"""
Phase 2: Clean real PM2.5 data, build leakage-free features and forecast origins.

Key design choices
- Only real measurements are used. Missing values stay missing in the targets
  (an `observed` mask) so models are never trained or scored on imputed values.
- Input gaps are filled causally (forward-fill only, then station mean), so no
  future value can leak into an input window. A binary `observed` feature tells
  the model which inputs were real.
- Per-station standardisation, fitted on the training period only.
- Instead of materialising millions of windows, we store the full [T, N, F] tensor
  and the list of valid forecast origins per split; windows are sliced on the fly.
"""
import json
import os

import numpy as np
import pandas as pd

import config as C


def clean_station_names(meta):
    name = meta["station_name"].str.strip()
    agency = name.str.extract(r"-\s*(\w+)\s*$")[0].fillna("")
    short = name.str.replace(r",?\s*Delhi\s*-\s*\w+\s*$", "", regex=True).str.strip()
    return short, agency


def flatline_mask(df, min_run=12):
    """True where a sensor repeats the exact same value >= min_run hours (stuck sensor)."""
    out = pd.DataFrame(False, index=df.index, columns=df.columns)
    for col in df.columns:
        s = df[col]
        grp = (s != s.shift()).cumsum()
        run = s.groupby(grp).transform("size")
        out[col] = s.notna() & (run >= min_run)
    return out


def main():
    print("=" * 70)
    print(f" Phase 2: Preprocessing real Delhi PM2.5 data ({C.START[:7]} .. {C.END[:7]})")
    print("=" * 70)
    os.makedirs(C.PROCESSED_DIR, exist_ok=True)

    # XKDR first, then CPCB files; where both have an hour (identical values) keep XKDR.
    sources = [pd.read_csv(C.RAW_CSV)]
    if os.path.exists(C.RAW_CSV_CPCB):
        sources.append(pd.read_csv(C.RAW_CSV_CPCB))
    raw = pd.concat(sources, ignore_index=True)
    raw["t"] = pd.to_datetime(raw["collected_at"], format="ISO8601").dt.floor("h")
    raw = raw.drop_duplicates(["station_id", "t"])
    idx = pd.date_range(C.START, C.END, freq="h")
    pm = raw.pivot(index="t", columns="station_id", values="value").reindex(idx)

    # 1. Remove non-physical readings and stuck sensors.
    pm = pm.where((pm > 0) & (pm <= 1000))
    stuck = flatline_mask(pm)
    print(f"Removed {int(stuck.values.sum()):,} flat-lined readings (stuck sensors).")
    pm = pm.mask(stuck)

    # 2. Station selection by real-data coverage.
    coverage = pm.loc[:C.SELECTION_END].notna().mean()
    keep = coverage[coverage >= C.MIN_COVERAGE].index
    dropped = sorted(set(pm.columns) - set(keep))
    print(f"Keeping {len(keep)} stations (coverage >= {C.MIN_COVERAGE:.0%}); dropped: {dropped}")

    meta = pd.read_csv(C.METADATA_CSV)
    meta = meta[meta["station_id"].isin(keep)].copy()
    meta["name"], meta["agency"] = clean_station_names(meta)
    meta = meta.rename(columns={"latitude": "lat", "longitude": "lon"})
    meta["coverage"] = meta["station_id"].map(coverage).round(3)
    meta = meta.sort_values("name").reset_index(drop=True)
    stations = meta[["station_id", "name", "agency", "lat", "lon", "coverage"]]
    stations.to_csv(C.STATIONS_CSV, index=False)
    ids = stations["station_id"].tolist()
    pm = pm[ids]

    T, N = pm.shape
    observed = pm.notna().values
    print(f"Hourly grid: {T:,} hours x {N} stations; real-data fraction = {observed.mean():.1%}")

    # 3. Chronological split boundaries (by time index).
    train_end = idx.get_loc(pd.Timestamp(C.TRAIN_END)) + 1
    val_end = idx.get_loc(pd.Timestamp(C.VAL_END)) + 1

    # 4. Per-station scaler fitted on training period only.
    train_vals = pm.iloc[:train_end]
    mean = train_vals.mean().values.astype(np.float32)
    std = train_vals.std().values.astype(np.float32)
    norm = (pm.values - mean) / std

    # 5. Causal gap filling for INPUTS only: forward-fill up to 24h, then station mean (0).
    filled = pd.DataFrame(norm, index=idx).ffill(limit=24).fillna(0.0).values

    # 6. Calendar features.
    hour, dow, doy = idx.hour.values, idx.day_of_week.values, idx.day_of_year.values
    cal = np.stack([np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                    np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
                    np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)], axis=1)
    features = np.zeros((T, N, len(C.FEATURE_NAMES)), dtype=np.float32)
    features[:, :, 0] = filled
    features[:, :, 1] = observed
    features[:, :, 2:] = cal[:, None, :]

    # 7. Valid forecast origins t (last input hour). All targets must lie in the same split
    #    and the input window must contain mostly real data (skips network-wide outages).
    W, Hmax = C.INPUT_WINDOW, max(C.HORIZONS)
    obs_frac_hour = observed.mean(axis=1)
    win_obs = pd.Series(obs_frac_hour).rolling(W).mean().values
    tgt_obs = np.zeros(T)
    for h in C.HORIZONS:
        tgt_obs += np.concatenate([obs_frac_hour[h:], np.zeros(h)])

    def origins(lo, hi):
        t = np.arange(max(lo, W - 1), hi - Hmax)
        ok = (win_obs[t] >= 0.5) & (obs_frac_hour[t] >= 0.5) & (tgt_obs[t] > 0)
        return t[ok].astype(np.int32)

    splits = {"train": origins(0, train_end), "val": origins(train_end, val_end),
              "test": origins(val_end, T)}
    print(f"Run: {C.RUN}")
    for k, v in splits.items():
        span = f"{idx[v[0]]} -> {idx[v[-1]]}" if len(v) else "(none)"
        print(f"  {k:5s}: {len(v):6,} origins  {span}")

    np.savez_compressed(
        C.DATASET_NPZ,
        features=features,
        pm25=pm.values.astype(np.float32),       # raw µg/m³ with NaN where missing
        observed=observed,
        times=idx.values.astype("datetime64[h]").astype(np.int64),
        train_idx=splits["train"], val_idx=splits["val"], test_idx=splits["test"],
    )
    with open(C.SCALER_JSON, "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist(), "station_ids": ids,
                   "input_window": W, "horizons": C.HORIZONS,
                   "features": C.FEATURE_NAMES, "train_end": C.TRAIN_END,
                   "val_end": C.VAL_END}, f, indent=2)
    print(f"Saved {C.DATASET_NPZ} and {C.SCALER_JSON}")


if __name__ == "__main__":
    main()
