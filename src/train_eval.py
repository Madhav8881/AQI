"""
Phase 4: Train LSTM, Transformer and GAT-Transformer with several random seeds,
then run the full evaluation (evaluate.py).

- Masked Huber loss: missing (unobserved) target hours contribute nothing.
- Early stopping on validation loss; best checkpoint per (model, seed) is kept.
- Each epoch uses a fresh random fraction of training origins (adjacent hourly windows are
  nearly identical, so this cuts CPU time with little information loss).
Usage:  python train_eval.py [--seeds 0 1 2] [--epochs 20] [--models LSTM ...] [--threads 4]
        python train_eval.py --eval-only
"""
import argparse
import json
import os
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


def run_epoch(model, data, origins, adj, opt=None, bs=64, frac=1.0, station_dropout=0.0):
    train = opt is not None
    model.train(train)
    total, count = 0.0, 0
    order = origins
    if train:
        order = np.random.permutation(origins)[: int(len(origins) * frac)]
    with torch.set_grad_enabled(train):
        for i in range(0, len(order), bs):
            X, Y, M = data.batch(order[i:i + bs])
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


def train_one(name, seed, data, epochs, patience, frac, station_dropout=0.0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(name, data.n_nodes, len(C.HORIZONS), window=C.INPUT_WINDOW)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    best, bad, hist = float("inf"), 0, {"train": [], "val": []}
    t0 = time.time()
    for ep in range(1, epochs + 1):
        tr = run_epoch(model, data, data.splits["train"], data.adj, opt, frac=frac,
                       station_dropout=station_dropout)
        va = run_epoch(model, data, data.splits["val"], data.adj, bs=256)
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
    return {"station_dropout": station_dropout, "best_val": best, "epochs": len(hist["train"]), "seconds": round(time.time() - t0, 1),
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
                res = train_one(name, seed, data, args.epochs, args.patience, args.train_frac,
                                args.station_dropout if args.station_dropout is not None
                                else STATION_DROPOUT.get(name, 0.0))
                with open(log_path(name, seed), "w") as f:
                    json.dump(res, f, indent=1)
    if not args.no_eval:
        evaluate.main()


if __name__ == "__main__":
    main()
