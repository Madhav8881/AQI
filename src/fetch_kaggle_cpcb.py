"""
Phase 1b: Download newer CPCB hourly data (2025-2026) for our stations from the public Kaggle
dataset "Delhi Air Quality: CPCB Station Data (2010-2026)" by Yug Kaushik
(kaggle.com/datasets/yug201/delhi-air-quality-cpcb-hourly-2010-2026), which mirrors the CPCB portal.

Only PM2.5 is kept. Output: data/raw/delhi_pm25_cpcb_2025_2026.csv
(columns: station_id, collected_at, value), same layout as the XKDR file.
Full per-station downloads are cached in data/raw/kaggle_cpcb/ (git-ignored).
"""
import json
import os
import time
import urllib.parse
import urllib.request

import pandas as pd

import config as C

DATASET = "yug201/delhi-air-quality-cpcb-hourly-2010-2026"
API = "https://www.kaggle.com/api/v1/datasets"
HEADERS = {"User-Agent": "aqi-research-client"}
YEARS = ["2025", "2026"]
CACHE = os.path.join(C.RAW_DIR, "kaggle_cpcb")
OUT_CSV = os.path.join(C.RAW_DIR, "delhi_pm25_cpcb_2025_2026.csv")


def list_files():
    files, token = [], None
    while True:
        url = f"{API}/list/{DATASET}?pageSize=200" + (f"&pageToken={token}" if token else "")
        with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=120) as r:
            d = json.loads(r.read())
        files += [f["name"] for f in d.get("datasetFiles", [])]
        token = d.get("nextPageToken")
        if not token:
            return files


def download(name):
    path = os.path.join(CACHE, name)
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    url = f"{API}/download/{DATASET}/{urllib.parse.quote(name, safe='')}"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=300) as r:
                data = r.read()
            break
        except Exception as e:  # transient network errors -> back off and retry
            if attempt == 4:
                raise
            print(f"  retry {attempt + 1} for {name}: {e}")
            time.sleep(3 * (attempt + 1))
    with open(path, "wb") as f:
        f.write(data)
    return path


def main():
    wanted = set(pd.read_csv(C.STATIONS_CSV)["station_id"])
    files = [f for f in list_files()
             if f.split("/")[0] in YEARS and "_".join(f.split("/")[1].split("_")[:2]) in wanted]
    print(f"Downloading {len(files)} station-year files ({', '.join(YEARS)})")
    frames = []
    for i, name in enumerate(sorted(files), 1):
        sid = "_".join(name.split("/")[1].split("_")[:2])
        df = pd.read_csv(download(name))
        pm_col = next(c for c in df.columns if c.startswith("PM2.5"))
        part = pd.DataFrame({"station_id": sid, "collected_at": df["Timestamp"], "value": df[pm_col]})
        frames.append(part.dropna(subset=["value"]))
        print(f"[{i:02d}/{len(files)}] {name}: {frames[-1].shape[0]:,} PM2.5 readings")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT_CSV, index=False)
    missing = sorted(wanted - set(out["station_id"]))
    print(f"Saved {len(out):,} readings to {OUT_CSV}; stations without data: {missing}")


if __name__ == "__main__":
    main()
