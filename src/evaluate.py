"""
Phase 5: Evaluate all models on the held-out test period using ONLY real observations.

Outputs (results/):
  metrics_overall.csv   mean ± std over seeds (MAE, RMSE, R², AQI-category accuracy)
  metrics_horizon.csv   per forecast horizon
  metrics_season.csv    per test year, 2026 only, winter vs summer, severe episodes
  metrics_station.csv   per station MAE (best seed)
  dm_test.csv           Diebold-Mariano test, GAT-Transformer vs each other model
  test_predictions.npz  test forecasts of every model (best-validation seed), µg/m³
"""
import json
import math
import os

import numpy as np
import pandas as pd
import torch

import config as C
from data import AQIData
from models import MODEL_CLASSES, Persistence, SeasonalNaive, build_model, slug

PROPOSED = "GAT-Transformer"
CAT_EDGES = np.array([b[1] for b in C.AQI_BANDS[:-1]])


def load_logs():
    """{model_name: {seed: training summary}} from results/train_logs/*.json."""
    log = {}
    for name in MODEL_CLASSES:
        prefix = slug(name) + "_seed"
        for fn in sorted(os.listdir(os.path.join(C.RESULTS_DIR, "train_logs"))):
            if fn.startswith(prefix) and fn.endswith(".json"):
                with open(os.path.join(C.RESULTS_DIR, "train_logs", fn)) as f:
                    log.setdefault(name, {})[fn[len(prefix):-5]] = json.load(f)
    return log


def load_model(name, seed, data):
    m = build_model(name, data.n_nodes, len(C.HORIZONS), window=C.INPUT_WINDOW)
    path = os.path.join(C.CHECKPOINTS_DIR, f"{slug(name)}_seed{seed}.pt")
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m.eval()


@torch.no_grad()
def predict(model, data, origins, bs=512):
    out = []
    for i in range(0, len(origins), bs):
        X, _, _ = data.batch(origins[i:i + bs])
        out.append(model(X, data.adj).numpy())
    return data.denorm(np.concatenate(out))                     # [S, H, N] µg/m³


def metrics(y, p, m):
    y, p = y[m], p[m]
    err = p - y
    ss_tot = ((y - y.mean()) ** 2).sum()
    return {"MAE": np.abs(err).mean(), "RMSE": np.sqrt((err ** 2).mean()),
            "R2": 1 - (err ** 2).sum() / ss_tot,
            "AQI_acc": (np.digitize(y, CAT_EDGES) == np.digitize(p, CAT_EDGES)).mean()}


def dm_test(e1, e2, h):
    """Diebold-Mariano test on absolute-error loss with Newey-West variance (lag h-1)."""
    d = np.abs(e1) - np.abs(e2)
    d = d[~np.isnan(d)]
    n, dbar = len(d), d.mean()
    gamma = [np.mean((d[k:] - dbar) * (d[:n - k] - dbar)) for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    stat = dbar / math.sqrt(max(var, 1e-12))
    p = math.erfc(abs(stat) / math.sqrt(2))
    return stat, p


def fmt(mean, std, nd=2):
    return f"{mean:.{nd}f} ± {std:.{nd}f}" if std > 0 else f"{mean:.{nd}f}"


def main():
    data = AQIData()
    origins = data.splits["test"]
    Y, M = data.targets_raw(origins)
    Y = np.nan_to_num(Y)
    times = data.times[origins]
    month = times.month
    seasons = {
        "Test year 1 (Oct 2024-Sep 2025)": times < pd.Timestamp("2025-10-01"),
        "Test year 2 (Oct 2025-Sep 2026)": times >= pd.Timestamp("2025-10-01"),
        "2026 only (Jan-Sep 2026)": times >= pd.Timestamp("2026-01-01"),
        "Winter smog (Oct-Feb)": np.isin(month, [10, 11, 12, 1, 2]),
        "Summer/Monsoon (Apr-Sep)": np.isin(month, [4, 5, 6, 7, 8, 9]),
    }
    log = load_logs()

    # Predictions per model per seed.
    preds = {}
    X_all = torch.cat([data.batch(origins[i:i + 2048])[0] for i in range(0, len(origins), 2048)])
    for name, base in [("Persistence", Persistence(C.HORIZONS)),
                       ("Seasonal naive", SeasonalNaive(C.HORIZONS, C.INPUT_WINDOW))]:
        preds[name] = {"-": data.denorm(base(X_all).numpy())}
    best_seed = {}
    for name in MODEL_CLASSES:
        if name not in log:
            continue
        preds[name] = {s: predict(load_model(name, int(s), data), data, origins) for s in log[name]}
        best_seed[name] = min(log[name], key=lambda s: log[name][s]["best_val"])

    def agg(fn):
        rows = []
        for name, by_seed in preds.items():
            vals = pd.DataFrame([fn(p) for p in by_seed.values()])
            rows.append((name, vals.mean(), vals.std(ddof=0).fillna(0)))
        return rows

    # Overall.
    overall = []
    for name, mu, sd in agg(lambda p: metrics(Y, p, M)):
        overall.append({"Model": name, "MAE": fmt(mu.MAE, sd.MAE), "RMSE": fmt(mu.RMSE, sd.RMSE),
                        "R²": fmt(mu.R2, sd.R2, 3), "AQI category acc.": fmt(mu.AQI_acc, sd.AQI_acc, 3),
                        "MAE_mean": mu.MAE, "MAE_std": sd.MAE, "RMSE_mean": mu.RMSE,
                        "R2_mean": mu.R2, "AQI_acc_mean": mu.AQI_acc})
    overall = pd.DataFrame(overall)

    # Per horizon.
    hrows = []
    for hi, h in enumerate(C.HORIZONS):
        for name, mu, sd in agg(lambda p: metrics(Y[:, hi], p[:, hi], M[:, hi])):
            hrows.append({"Model": name, "Horizon": h, "MAE_mean": mu.MAE, "MAE_std": sd.MAE,
                          "RMSE_mean": mu.RMSE, "RMSE_std": sd.RMSE, "R2_mean": mu.R2,
                          "AQI_acc_mean": mu.AQI_acc})
    horizon = pd.DataFrame(hrows)

    # Per season and severe episodes (observed PM2.5 > 250).
    srows = []
    for sname, smask in seasons.items():
        for name, mu, sd in agg(lambda p: metrics(Y[smask], p[smask], M[smask])):
            srows.append({"Subset": sname, "Model": name, "MAE_mean": mu.MAE, "MAE_std": sd.MAE,
                          "RMSE_mean": mu.RMSE, "R2_mean": mu.R2})
    sev = M & (Y > 250)
    for name, mu, sd in agg(lambda p: {"MAE": np.abs(p - Y)[sev].mean()}):
        srows.append({"Subset": "Severe episodes (>250 µg/m³)", "Model": name,
                      "MAE_mean": mu.MAE, "MAE_std": sd.MAE, "RMSE_mean": np.nan, "R2_mean": np.nan})
    season = pd.DataFrame(srows)

    # Best-seed predictions (for station metrics, DM test and the app).
    best = {n: (p[best_seed[n]] if n in best_seed else p["-"]) for n, p in preds.items()}
    st_rows = []
    for n, p in best.items():
        for j, sid in enumerate(data.stations["station_id"]):
            m = M[:, :, j]
            st_rows.append({"Model": n, "station_id": sid,
                            "MAE": np.abs(p[:, :, j] - Y[:, :, j])[m].mean()})
    station = pd.DataFrame(st_rows)

    dm_rows = []
    if PROPOSED in best:
        for n in best:
            if n == PROPOSED:
                continue
            for hi, h in enumerate(C.HORIZONS):
                err = np.where(M[:, hi], best[PROPOSED][:, hi] - Y[:, hi], np.nan)
                err2 = np.where(M[:, hi], best[n][:, hi] - Y[:, hi], np.nan)
                keep = M[:, hi].any(1)
                e1 = np.nanmean(np.abs(err[keep]), 1)
                e2 = np.nanmean(np.abs(err2[keep]), 1)
                stat, pv = dm_test(e1, e2, h)
                dm_rows.append({"Comparison": f"{PROPOSED} vs {n}", "Horizon": h,
                                "DM stat": round(stat, 2), "p-value": pv,
                                "Better": PROPOSED if stat < 0 else n})
    dm = pd.DataFrame(dm_rows)

    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    overall.to_csv(os.path.join(C.RESULTS_DIR, "metrics_overall.csv"), index=False)
    horizon.to_csv(os.path.join(C.RESULTS_DIR, "metrics_horizon.csv"), index=False)
    season.to_csv(os.path.join(C.RESULTS_DIR, "metrics_season.csv"), index=False)
    station.to_csv(os.path.join(C.RESULTS_DIR, "metrics_station.csv"), index=False)
    dm.to_csv(os.path.join(C.RESULTS_DIR, "dm_test.csv"), index=False)
    np.savez_compressed(os.path.join(C.RESULTS_DIR, "test_predictions.npz"), origins=origins,
                        names=np.array(list(best)), preds=np.stack([best[n] for n in best]))
    with open(os.path.join(C.RESULTS_DIR, "best_seeds.json"), "w") as f:
        json.dump(best_seed, f)

    pd.set_option("display.width", 200)
    print("\nOVERALL (test, real observations only, mean ± std over seeds)")
    print(overall[["Model", "MAE", "RMSE", "R²", "AQI category acc."]].to_string(index=False))
    print("\nMAE BY HORIZON")
    print(horizon.pivot(index="Model", columns="Horizon", values="MAE_mean").round(2).to_string())
    print("\nMAE BY SUBSET")
    print(season.pivot(index="Model", columns="Subset", values="MAE_mean").round(2).to_string())
    if len(dm):
        print("\nDIEBOLD-MARIANO (negative stat = GAT-Transformer better)")
        print(dm.to_string(index=False))


if __name__ == "__main__":
    main()
