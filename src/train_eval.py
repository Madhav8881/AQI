"""
Phase 4: Train LSTM, Transformer and GAT-Transformer with several random seeds,
then run the full evaluation (evaluate.py).

- Masked Huber loss: missing (unobserved) target hours contribute nothing.
- Early stopping on validation loss; best checkpoint per (model, seed) is kept.
- Each epoch uses a fresh random fraction of training origins (adjacent hourly windows are
  nearly identical, so this cuts CPU time with little information loss).
- Optional history dropout (--history-dropout p): with probability p a training sample keeps only
  its last k hours of input (k random in 1..23), so the model also works right after start-up in
  live mode, when only a few hours of readings have been collected. --finetune continues from the
  existing checkpoint instead of starting from scratch (the original is kept as *_fullhist.pt).
Usage:  python train_eval.py [--seeds 0 1 2] [--epochs 20] [--models LSTM ...] [--threads 4]
        python train_eval.py --models "GAT-Transformer" --finetune --history-dropout 0.5 --epochs 8
        python train_eval.py --models "GAT-Transformer" --train-on-all --history-dropout 0.5 --epochs 3
        python train_eval.py --eval-only
"""
import argparse
import json
import os
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

import config as C
import evaluate
from data import AQIData
from models import MODEL_CLASSES, STATION_DROPOUT, build_model, count_params, slug


def ckpt_path(name, seed):
    return os.path.join(C.CHECKPOINTS_DIR, f"{slug(name)}_seed{seed}.pt")


def masked_huber(pred, y, mask):
    loss = F.huber_loss(pred, y, reduction="none", delta=1.0)
    m = mask.float()
    return (loss * m).sum() / m.sum().clamp_min(1.0)


def truncate_history(X, keep):
    """Hide all but the last `keep` hours of PM2.5 (value and observed flag) per sample.
    X: [B, W, N, F]; keep: int or LongTensor [B]. Calendar features stay (they are always known)."""
    B, W = X.shape[0], X.shape[1]
    keep = torch.as_tensor(keep).expand(B) if torch.as_tensor(keep).dim() == 0 else torch.as_tensor(keep)
    hidden = (torch.arange(W)[None, :] < (W - keep)[:, None]).float()[:, :, None]   # [B, W, 1]
    X = X.clone()
    X[..., 0] *= 1 - hidden
    X[..., 1] *= 1 - hidden
    return X


def run_epoch(model, data, origins, adj, opt=None, bs=64, frac=1.0, station_dropout=0.0,
              history_dropout=0.0, keep_last=None):
    train = opt is not None
    model.train(train)
    total, count = 0.0, 0
    order = origins
    if train:
        order = np.random.permutation(origins)[: int(len(origins) * frac)]
    with torch.set_grad_enabled(train):
        for i in range(0, len(order), bs):
            X, Y, M = data.batch(order[i:i + bs])
            if keep_last is not None:
                X = truncate_history(X, keep_last)
            if train and history_dropout > 0:
                short = torch.rand(X.shape[0]) < history_dropout
                keep = torch.where(short, torch.randint(1, C.INPUT_WINDOW, (X.shape[0],)),
                                   torch.full((X.shape[0],), C.INPUT_WINDOW))
                X = truncate_history(X, keep)
            if train and station_dropout > 0:
                # Hide whole stations' input windows at random so the model learns to
                # estimate a station from its neighbours when its sensor fails.
                drop = (torch.rand(X.shape[0], 1, X.shape[2]) < station_dropout).float()
                X = X.clone()
                X[..., 0] *= 1 - drop
                X[..., 1] *= 1 - drop
            loss = masked_huber(model(X, adj), Y, M)
            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += loss.item() * len(X)
            count += len(X)
    return total / count


def log_path(name, seed):
    return os.path.join(C.RESULTS_DIR, "train_logs", f"{slug(name)}_seed{seed}.json")


def validate(model, data, history_dropout):
    """Validation loss on full history; when training for short history, averaged with the
    loss when only the current hour is available."""
    full = run_epoch(model, data, data.splits["val"], data.adj, bs=256)
    if history_dropout <= 0:
        return full
    short = run_epoch(model, data, data.splits["val"], data.adj, bs=256, keep_last=1)
    return 0.5 * (full + short)


def train_on_all(name, seed, data, epochs, frac, station_dropout=0.0, history_dropout=0.0):
    """Final step for deployment: continue from the current checkpoint on ALL data (train + the
    months that were only used for early stopping), for a fixed number of epochs with a decaying
    learning rate. The per-station scaler is left unchanged so the checkpoint stays valid."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    path = ckpt_path(name, seed)
    backup = path.replace(".pt", "_beforeall.pt")
    if not os.path.exists(backup):
        shutil.copyfile(path, backup)
    model = build_model(name, data.n_nodes, len(C.HORIZONS), window=C.INPUT_WINDOW)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    origins = np.concatenate([data.splits["train"], data.splits["val"]])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    hist, t0 = {"train": []}, time.time()
    print(f"  [{name} s{seed}] training on all {len(origins):,} origins "
          f"({data.times[origins[0]]} -> {data.times[origins[-1]]})", flush=True)
    for ep in range(1, epochs + 1):
        tr = run_epoch(model, data, origins, data.adj, opt, frac=frac,
                       station_dropout=station_dropout, history_dropout=history_dropout)
        sched.step()
        hist["train"].append(tr)
        print(f"  [{name} s{seed}] ep {ep:02d} train {tr:.4f} ({time.time() - t0:.0f}s)", flush=True)
    torch.save(model.state_dict(), path)
    return {"trained_on_all": True, "origins": int(len(origins)), "station_dropout": station_dropout,
            "history_dropout": history_dropout, "epochs": epochs, "seconds": round(time.time() - t0, 1),
            "params": count_params(model), "history": hist}


def train_one(name, seed, data, epochs, patience, frac, station_dropout=0.0,
              history_dropout=0.0, finetune=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(name, data.n_nodes, len(C.HORIZONS), window=C.INPUT_WINDOW)
    lr = 1e-3
    best = float("inf")
    if finetune:
        path = ckpt_path(name, seed)
        backup = path.replace(".pt", "_fullhist.pt")
        if not os.path.exists(backup):
            shutil.copyfile(path, backup)            # keep the original full-history model
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        lr = 3e-4
        best = validate(model, data, history_dropout)  # only save if fine-tuning beats this
        print(f"  [{name} s{seed}] starting from {os.path.basename(path)}: val {best:.4f}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    bad, hist = 0, {"train": [], "val": []}
    t0 = time.time()
    for ep in range(1, epochs + 1):
        tr = run_epoch(model, data, data.splits["train"], data.adj, opt, frac=frac,
                       station_dropout=station_dropout, history_dropout=history_dropout)
        va = validate(model, data, history_dropout)
        sched.step(va)
        hist["train"].append(tr)
        hist["val"].append(va)
        flag = ""
        if va < best:
            best, bad, flag = va, 0, " *"
            torch.save(model.state_dict(), ckpt_path(name, seed))
        else:
            bad += 1
        print(f"  [{name} s{seed}] ep {ep:02d} train {tr:.4f} val {va:.4f} "
              f"({time.time() - t0:.0f}s){flag}", flush=True)
        if bad >= patience:
            break
    return {"station_dropout": station_dropout, "history_dropout": history_dropout,
            "finetuned": finetune, "best_val": best, "epochs": len(hist["train"]), "seconds": round(time.time() - t0, 1),
            "params": count_params(model), "history": hist}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--station-dropout", type=float, default=None,
                    help="override: probability of hiding a station's whole input window during "
                         "training (default: 0.15 for 'Robust ...' models, else 0)")
    ap.add_argument("--history-dropout", type=float, default=0.0,
                    help="probability that a training sample keeps only its last 1-23 input hours")
    ap.add_argument("--finetune", action="store_true",
                    help="continue from the existing checkpoint (lr 3e-4) instead of training from scratch")
    ap.add_argument("--train-on-all", action="store_true",
                    help="final deployment step: continue from the checkpoint on train + validation "
                         "months together for a fixed number of --epochs (no early stopping)")
    ap.add_argument("--models", nargs="+", default=list(MODEL_CLASSES))
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    if not args.eval_only:
        os.makedirs(C.CHECKPOINTS_DIR, exist_ok=True)
        os.makedirs(os.path.join(C.RESULTS_DIR, "train_logs"), exist_ok=True)
        data = AQIData()
        print(f"Train/val/test origins: {[len(v) for v in data.splits.values()]}, "
              f"stations: {data.n_nodes}, threads: {torch.get_num_threads()}")
        for name in args.models:
            for seed in args.seeds:
                print(f"\n=== {name} | seed {seed} ===", flush=True)
                sd = (args.station_dropout if args.station_dropout is not None
                      else STATION_DROPOUT.get(name, 0.0))
                if args.train_on_all:
                    res = train_on_all(name, seed, data, args.epochs, args.train_frac, sd,
                                       args.history_dropout)
                    out_path = log_path(name, seed).replace(".json", "_all.json")
                else:
                    res = train_one(name, seed, data, args.epochs, args.patience, args.train_frac,
                                    sd, args.history_dropout, args.finetune)
                    out_path = log_path(name, seed)
                with open(out_path, "w") as f:
                    json.dump(res, f, indent=1)
    if not args.no_eval:
        evaluate.main()


if __name__ == "__main__":
    main()
