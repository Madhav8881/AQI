"""
Live mode: current PM2.5 for our 36 stations from the World Air Quality Index project (WAQI,
aqicn.org — which republishes CPCB/DPCC station data), stored hour by hour, plus live forecasts.

- Stations are matched to WAQI stations by name (within 5 km), else by location (within 1 km).
- WAQI reports PM2.5 as a US-EPA (2016 scale) sub-index; it is converted back to µg/m³.
  The scale tops out at AQI 500 (≈ 500 µg/m³), so readings above that are capped.
- Every collection stores readings in data/live/readings.csv and, once per hour, issues a
  forecast from every model into data/live/forecasts.csv. Forecasts are later scored against
  the readings that actually arrive — a genuine live accuracy check.

Usage:  python live.py            # collect once
        python live.py --loop     # keep collecting every 15 minutes
Needs WAQI_TOKEN in .env (free personal token from aqicn.org/data-platform/token).
"""
import argparse
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch

# Use the final models (trained on all data up to Sep 2026) for live forecasts once they exist.
if "AQI_RUN" not in os.environ and os.path.exists(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results", "final", "best_seeds.json")):
    os.environ["AQI_RUN"] = "final"

import config as C  # noqa: E402

LIVE_DIR = os.path.join(C.ROOT, "data", "live")
READINGS_CSV = os.path.join(LIVE_DIR, "readings.csv")
FORECASTS_CSV = os.path.join(LIVE_DIR, "forecasts.csv" if C.RUN == "holdout" else f"forecasts_{C.RUN}.csv")
MAP_CSV = os.path.join(C.PROCESSED_DIR, "waqi_station_map.csv")
DELHI_BOUNDS = "28.40,76.85,28.90,77.40"
MAX_MATCH_KM = 1.0       # renamed stations: must be at (almost) the same spot
MAX_NAME_MATCH_KM = 5.0  # same-name stations: WAQI coordinates can be a few km off

# US-EPA 2016 PM2.5 breakpoints used by aqicn: (AQI low, AQI high, conc low, conc high).
EPA2016_PM25 = [(0, 50, 0.0, 12.0), (51, 100, 12.1, 35.4), (101, 150, 35.5, 55.4),
                (151, 200, 55.5, 150.4), (201, 300, 150.5, 250.4), (301, 400, 250.5, 350.4),
                (401, 500, 350.5, 500.4)]


def aqi_to_pm25(aqi):
    """Invert the US-EPA (2016) PM2.5 sub-index to a concentration in µg/m³."""
    if aqi is None or not np.isfinite(aqi):
        return np.nan
    aqi = min(max(float(aqi), 0.0), 500.0)
    for a_lo, a_hi, c_lo, c_hi in EPA2016_PM25:
        if aqi <= a_hi:
            return c_lo + (aqi - a_lo) * (c_hi - c_lo) / (a_hi - a_lo)
    return 500.4


def load_token():
    tok = os.environ.get("WAQI_TOKEN")
    env = os.path.join(C.ROOT, ".env")
    if not tok and os.path.exists(env):
        with open(env, encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("WAQI_TOKEN="):
                    tok = line.split("=", 1)[1].strip()
    return tok or None


def _get(url, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "aqi-research-client"})
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read())
            if d.get("status") != "ok":
                raise RuntimeError(d.get("data"))
            return d["data"]
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def station_map(token, rebuild=False):
    """Our station_id -> WAQI uid, matched by location. Cached in data/processed."""
    if os.path.exists(MAP_CSV) and not rebuild:
        return pd.read_csv(MAP_CSV)
    waqi = pd.DataFrame(_get(f"https://api.waqi.info/map/bounds?latlng={DELHI_BOUNDS}"
                             f"&networks=all&token={token}"))
    waqi["waqi_name"] = waqi["station"].map(lambda s: s["name"])
    ours = pd.read_csv(C.STATIONS_CSV)
    squash = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())
    waqi_key = waqi["waqi_name"].map(squash)
    rows, used = [], set()
    for _, s in ours.iterrows():
        d = _haversine_km(s.lat, s.lon, waqi["lat"].values, waqi["lon"].values)
        # 1) Same station name (WAQI coordinates can be a few km off) ...
        cand = [j for j in np.argsort(d) if squash(s["name"]) in waqi_key[j]
                and d[j] <= MAX_NAME_MATCH_KM and waqi.uid[j] not in used]
        # 2) ... otherwise a renamed station at (almost) the same spot.
        if not cand:
            cand = [j for j in np.argsort(d) if d[j] <= MAX_MATCH_KM and waqi.uid[j] not in used]
        j = cand[0] if cand else None
        if j is not None:
            used.add(waqi.uid[j])
        rows.append({"station_id": s.station_id, "name": s["name"],
                     "waqi_uid": int(waqi.uid[j]) if j is not None else None,
                     "waqi_name": waqi.waqi_name[j] if j is not None else None,
                     "distance_km": round(float(d[j]), 2) if j is not None else None})
    m = pd.DataFrame(rows)
    m.to_csv(MAP_CSV, index=False)
    return m


def fetch_current(token):
    """Latest PM2.5 reading per station: DataFrame[station_id, time, pm25_aqi, pm25]."""
    m = station_map(token).dropna(subset=["waqi_uid"])

    def one(row):
        try:
            d = _get(f"https://api.waqi.info/feed/@{int(row.waqi_uid)}/?token={token}")
            aqi = d.get("iaqi", {}).get("pm25", {}).get("v")
            t = pd.Timestamp(d["time"]["s"]).floor("h")          # local IST, naive
            return {"station_id": row.station_id, "time": t, "pm25_aqi": aqi,
                    "pm25": round(aqi_to_pm25(aqi), 1) if aqi is not None else np.nan}
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=6) as pool:
        out = [r for r in pool.map(one, m.itertuples()) if r]
    return pd.DataFrame(out)


def load_readings():
    if not os.path.exists(READINGS_CSV):
        return pd.DataFrame(columns=["station_id", "time", "pm25_aqi", "pm25"])
    df = pd.read_csv(READINGS_CSV, parse_dates=["time"])
    return df


def load_forecasts():
    if not os.path.exists(FORECASTS_CSV):
        return pd.DataFrame(columns=["issued_at", "model", "station_id", "horizon", "target_time", "pred"])
    return pd.read_csv(FORECASTS_CSV, parse_dates=["issued_at", "target_time"])


def store_readings(new):
    os.makedirs(LIVE_DIR, exist_ok=True)
    df = pd.concat([load_readings(), new], ignore_index=True)
    df = df.dropna(subset=["pm25"]).drop_duplicates(["station_id", "time"], keep="last")
    df.sort_values(["time", "station_id"]).to_csv(READINGS_CSV, index=False)
    return df


def build_window(readings, data, end):
    """Model input [1, 24, N, F] for the window ending at hour `end`, built exactly like
    preprocess.py: per-station scaling, causal forward-fill (<= 24 h) then station mean, an
    observed flag, and calendar features. Returns (X, hours_with_data)."""
    ids = data.stations["station_id"].tolist()
    idx = pd.date_range(end - pd.Timedelta(hours=47), end, freq="h")
    pm = readings.pivot_table(index="time", columns="station_id", values="pm25", aggfunc="last")
    pm = pm.reindex(index=idx, columns=ids)
    observed = pm.notna().values
    norm = (pm.values - data.mean) / data.std
    filled = pd.DataFrame(norm, index=idx).ffill(limit=24).fillna(0.0).values
    hour, dow, doy = idx.hour.values, idx.day_of_week.values, idx.day_of_year.values
    cal = np.stack([np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                    np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
                    np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)], axis=1)
    F = np.zeros((len(idx), len(ids), len(C.FEATURE_NAMES)), dtype=np.float32)
    F[:, :, 0], F[:, :, 1], F[:, :, 2:] = filled, observed, cal[:, None, :]
    X = torch.from_numpy(F[-C.INPUT_WINDOW:]).unsqueeze(0)
    hours_with_data = int(observed[-C.INPUT_WINDOW:].any(axis=1).sum())
    return X, hours_with_data


# Average of these models' forecasts: ~1% lower error than the best single model on the
# two-year held-out test (26.54 vs 26.77 µg/m³ MAE).
ENSEMBLE_MEMBERS = ["GAT-Transformer", "Robust GAT-Transformer", "Transformer"]


def add_ensemble(out):
    members = [out[m] for m in ENSEMBLE_MEMBERS if m in out]
    if len(members) == len(ENSEMBLE_MEMBERS):
        out["Ensemble"] = np.mean(members, axis=0)
    return out


@torch.no_grad()
def forecast(models, data, readings, end):
    from models import Persistence
    X, n_hours = build_window(readings, data, end)
    out = {"Persistence": data.denorm(Persistence(C.HORIZONS)(X).numpy()[0])}
    for name, m in models.items():
        out[name] = data.denorm(m(X, data.adj).numpy()[0])
    return add_ensemble(out), n_hours


def log_forecast(fc, data, issued_at, history_hours):
    old = load_forecasts()
    if len(old) and (old["issued_at"] == issued_at).any():
        return old
    rows = []
    for name, arr in fc.items():
        for hi, h in enumerate(C.HORIZONS):
            for j, sid in enumerate(data.stations["station_id"]):
                rows.append({"issued_at": issued_at, "history_hours": history_hours,
                             "model": name, "station_id": sid, "horizon": h,
                             "target_time": issued_at + pd.Timedelta(hours=h),
                             "pred": round(float(arr[hi, j]), 1)})
    df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(FORECASTS_CSV, index=False)
    return df


MIN_HISTORY_FOR_SCORING = 1


def live_scores(readings, forecasts, min_history=MIN_HISTORY_FOR_SCORING):
    """MAE of logged live forecasts whose target hour has since been observed. Forecasts issued with
    fewer than min_history hours of collected input are skipped. Accuracy is a little lower when little
    history has been collected (about 12% higher error with only the current hour)."""
    if forecasts.empty or readings.empty:
        return pd.DataFrame()
    hist = forecasts["history_hours"] if "history_hours" in forecasts else pd.Series(0, index=forecasts.index)
    forecasts = forecasts[hist.fillna(0) >= min_history]
    j = forecasts.merge(readings[["station_id", "time", "pm25"]],
                        left_on=["station_id", "target_time"], right_on=["station_id", "time"])
    if j.empty:
        return pd.DataFrame()
    j["abs_err"] = (j["pred"] - j["pm25"]).abs()
    return j.groupby(["model", "horizon"]).agg(MAE=("abs_err", "mean"), n=("abs_err", "size")).reset_index()


def load_models(data):
    from evaluate import load_model
    best = json.load(open(os.path.join(C.RESULTS_DIR, "best_seeds.json")))
    return {n: load_model(n, int(s), data) for n, s in best.items()}


def collect(data=None, models=None, token=None):
    """Fetch current readings, store them, and log a forecast for the latest hour."""
    from data import AQIData
    token = token or load_token()
    if not token:
        raise SystemExit("Add WAQI_TOKEN=... to .env (free token: aqicn.org/data-platform/token)")
    data = data or AQIData()
    models = models or load_models(data)
    new = fetch_current(token)
    readings = store_readings(new)
    latest = new["time"].max() if len(new) else readings["time"].max()
    fc, n_hours = forecast(models, data, readings, latest)
    log_forecast(fc, data, latest, n_hours)
    return {"stations": int(new["pm25"].notna().sum()) if len(new) else 0,
            "latest": latest, "hours_of_history": n_hours}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true", help="collect every --every minutes")
    ap.add_argument("--every", type=int, default=15)
    args = ap.parse_args()
    from data import AQIData
    data = AQIData()
    models = load_models(data)
    while True:
        try:
            r = collect(data, models)
            print(f"{pd.Timestamp.now():%Y-%m-%d %H:%M} collected {r['stations']} stations, "
                  f"latest hour {r['latest']}, history {r['hours_of_history']}/24 h", flush=True)
        except Exception as e:
            print(f"{pd.Timestamp.now():%Y-%m-%d %H:%M} collection failed: {e}", flush=True)
        if not args.loop:
            break
        time.sleep(args.every * 60)


if __name__ == "__main__":
    main()
