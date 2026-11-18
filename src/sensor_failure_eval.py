"""
Phase 6: Sensor-failure test ("what if a station's sensor breaks?").

For every station j, we hide j's last 24 hours of readings from the models and ask them to
forecast j anyway. Models that look at each station in isolation (LSTM, Transformer) have
nothing to go on; the GAT-Transformer can still use neighbouring stations.
Errors are measured only on real observed values in the test period.

Output: results/sensor_failure.csv            (model x horizon: MAE normal vs sensor failed)
        results/sensor_failure_station.csv    (model x station MAE with sensor failed)
Usage:  python sensor_failure_eval.py [--stride 3]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

import config as C
from data import AQIData
from evaluate import load_model
from models import MODEL_CLASSES, Persistence


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=3, help="use every k-th test origin")
    args = ap.parse_args()

    data = AQIData()
    origins = data.splits["test"][:: args.stride]
    Y, M = data.targets_raw(origins)
    Y = np.nan_to_num(Y)
    X = torch.cat([data.batch(origins[i:i + 2048])[0] for i in range(0, len(origins), 2048)])
    best = json.load(open(os.path.join(C.RESULTS_DIR, "best_seeds.json")))

    runners = {"Station average": lambda x: Persistence(C.HORIZONS)(x)}  # hidden input = mean
    for name in MODEL_CLASSES:
        if name in best:
            m = load_model(name, int(best[name]), data)
            runners[name] = lambda x, m=m: torch.cat([m(x[i:i + 512], data.adj)
                                                      for i in range(0, len(x), 512)])

    rows, st_rows = [], []
    for name, run in runners.items():
        normal = data.denorm(run(X).numpy()) if name != "Station average" else None
        failed = np.zeros_like(Y)
        for j in range(data.n_nodes):
            failed[:, :, j] = data.denorm(run(data.hide_stations(X, [j])).numpy())[:, :, j]
            st_rows.append({"Model": name, "station_id": data.stations.station_id[j],
                            "MAE_failed": np.abs(failed[:, :, j] - Y[:, :, j])[M[:, :, j]].mean()})
        for hi, h in enumerate(C.HORIZONS):
            m = M[:, hi]
            rows.append({"Model": name, "Horizon": h,
                         "MAE_normal": np.abs(normal[:, hi] - Y[:, hi])[m].mean() if normal is not None else np.nan,
                         "MAE_failed": np.abs(failed[:, hi] - Y[:, hi])[m].mean()})
        print(f"done: {name}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.RESULTS_DIR, "sensor_failure.csv"), index=False)
    pd.DataFrame(st_rows).to_csv(os.path.join(C.RESULTS_DIR, "sensor_failure_station.csv"), index=False)
    print("\nMAE (µg/m³) when the station's own sensor has failed:")
    print(df.pivot(index="Model", columns="Horizon", values="MAE_failed").round(2).to_string())
    print("\nMAE (µg/m³) with the sensor working (same origins):")
    print(df.pivot(index="Model", columns="Horizon", values="MAE_normal").round(2).to_string())


if __name__ == "__main__":
    main()
