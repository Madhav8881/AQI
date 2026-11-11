"""Shared configuration for the Delhi PM2.5 spatiotemporal forecasting pipeline."""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RAW_DIR = os.path.join(ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
CHECKPOINTS_DIR = os.path.join(ROOT, "checkpoints")
RESULTS_DIR = os.path.join(ROOT, "results")

RAW_CSV = os.path.join(RAW_DIR, "delhi_pm25_2023_2025_raw.csv")
METADATA_CSV = os.path.join(RAW_DIR, "delhi_stations_metadata.csv")
STATIONS_CSV = os.path.join(PROCESSED_DIR, "stations.csv")
DATASET_NPZ = os.path.join(PROCESSED_DIR, "dataset.npz")
SCALER_JSON = os.path.join(PROCESSED_DIR, "scaler_params.json")
ADJ_NPY = os.path.join(PROCESSED_DIR, "adjacency_matrix.npy")
DIST_NPY = os.path.join(PROCESSED_DIR, "distance_matrix.npy")

# Timeline (real measurements only). Jan-Mar 2025 is missing network-wide in the source.
START = "2023-01-01 00:00:00"
END = "2025-08-31 23:00:00"
TRAIN_END = "2024-06-30 23:00:00"   # train: Jan 2023 - Jun 2024
VAL_END = "2024-09-30 23:00:00"     # val:   Jul 2024 - Sep 2024; test: Oct 2024 - Aug 2025

MIN_COVERAGE = 0.70                 # drop stations with less real data than this
INPUT_WINDOW = 24                   # hours of history fed to the models
HORIZONS = [1, 3, 6, 12, 24]        # forecast lead times (hours)
KNN = 8                             # graph neighbours per station
FEATURE_NAMES = ["pm25", "observed", "hour_sin", "hour_cos",
                 "dow_sin", "dow_cos", "doy_sin", "doy_cos"]

# Indian National AQI (CPCB) PM2.5 24h breakpoints, used for category colouring/accuracy.
AQI_BANDS = [
    (0, 30, "Good", "#00B050"),
    (30, 60, "Satisfactory", "#92D050"),
    (60, 90, "Moderate", "#FFD700"),
    (90, 120, "Poor", "#FF9900"),
    (120, 250, "Very Poor", "#FF0000"),
    (250, 10000, "Severe", "#7E0023"),
]


def aqi_category(v):
    for lo, hi, name, _ in AQI_BANDS:
        if v < hi:
            return name
    return "Severe"
