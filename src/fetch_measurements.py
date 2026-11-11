"""
Phase 1: Fetch REAL hourly PM2.5 measurements (no synthetic data) for all Delhi
CAAQM stations from the XKDR Air Quality API, 2023-01-01 to 2025-08-31.

- Requests are chunked by quarter (API returns max 10,000 rows per call).
- Each station is cached to data/raw/stations/<station_id>.csv so reruns resume.
- API key is read from the XKDR_API_KEY environment variable or a local .env file.
"""
import os
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RAW_DIR = os.path.join(ROOT, "data", "raw")
STATION_CACHE = os.path.join(RAW_DIR, "stations")
os.makedirs(STATION_CACHE, exist_ok=True)

START, END = "2023-01-01", "2025-08-31"
OUT_CSV = os.path.join(RAW_DIR, "delhi_pm25_2023_2025_raw.csv")

# Duplicate sites sharing identical coordinates, or a different (half-hourly) source.
EXCLUDE = {"site_107", "site_5395", "DS1010001"}


def load_api_key():
    key = os.environ.get("XKDR_API_KEY")
    env_path = os.path.join(ROOT, ".env")
    if not key and os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("XKDR_API_KEY="):
                    key = line.split("=", 1)[1].strip()
    if not key or key == "your_api_key_here":
        raise SystemExit("Set XKDR_API_KEY in your environment or in .env (see .env.example).")
    return key


def quarter_ranges(start, end):
    edges = list(pd.date_range(start, end, freq="QS"))
    if not edges or edges[0] > pd.Timestamp(start):
        edges.insert(0, pd.Timestamp(start))
    out = []
    for i, s in enumerate(edges):
        e = edges[i + 1] - pd.Timedelta(days=1) if i + 1 < len(edges) else pd.Timestamp(end)
        out.append((s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")))
    return out


def fetch_chunk(headers, station_id, start, end, retries=3):
    url = (f"https://airquality.xkdr.org/v1/measurements?station={station_id}"
           f"&parameter=PM2.5&start={start}&end={end}")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if payload.get("meta", {}).get("truncated"):
                print(f"  [warn] truncated response {station_id} {start}..{end}")
            return payload.get("data", [])
        except Exception as e:  # network hiccup -> back off and retry
            if attempt == retries - 1:
                print(f"  [error] {station_id} {start}..{end}: {e}")
                return []
            time.sleep(2 * (attempt + 1))


def fetch_station(headers, station_id):
    cache = os.path.join(STATION_CACHE, f"{station_id}.csv")
    if os.path.exists(cache):
        return station_id, pd.read_csv(cache)
    rows = []
    for s, e in quarter_ranges(START, END):
        rows.extend(fetch_chunk(headers, station_id, s, e))
        time.sleep(0.2)
    df = pd.DataFrame(rows, columns=["station_id", "collected_at", "value"]) if rows else \
        pd.DataFrame(columns=["station_id", "collected_at", "value"])
    df = df[["station_id", "collected_at", "value"]]
    df.to_csv(cache, index=False)
    return station_id, df


def main():
    headers = {"Authorization": f"Bearer {load_api_key()}", "User-Agent": "aqi-research-client"}
    meta = pd.read_csv(os.path.join(RAW_DIR, "delhi_stations_metadata.csv"))
    meta = meta[~meta["station_id"].isin(EXCLUDE)]
    ids = meta["station_id"].tolist()
    print(f"Fetching real PM2.5 for {len(ids)} Delhi stations, {START} .. {END}")

    frames = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch_station, headers, sid) for sid in ids]
        for i, fut in enumerate(as_completed(futures), 1):
            sid, df = fut.result()
            print(f"[{i:02d}/{len(ids)}] {sid}: {len(df):,} rows")
            frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(all_df):,} real measurements to {OUT_CSV}")


if __name__ == "__main__":
    main()
