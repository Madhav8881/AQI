"""
Phase 2: Data Preprocessing, Cleaning, Imputation, and Sliding Window Sequence Generation.
Generates chronological Train (70%), Val (15%), and Test (15%) splits with 24h lookback and multi-horizon targets.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

INPUT_WINDOW = 24 # 24 hours lookback
HORIZONS = [1, 3, 6, 12] # Multi-horizon forecast targets (+1h, +3h, +6h, +12h)

def preprocess_air_quality_data():
    print("=" * 65)
    print(" Phase 2: Preprocessing 2023-2024 Delhi Air Quality Records ")
    print("=" * 65)
    
    # 1. Load stations and raw measurement tables
    stations_df = pd.read_csv(os.path.join(RAW_DIR, "selected_stations.csv"))
    station_ids = stations_df['station_id'].tolist()
    n_stations = len(station_ids)
    print(f"Target Stations ({n_stations}): {station_ids}")
    
    raw_csv_path = os.path.join(RAW_DIR, "delhi_pm25_2023_2026_raw.csv")
    if not os.path.exists(raw_csv_path):
        raw_csv_path = os.path.join(RAW_DIR, "delhi_pm25_2023_2024_raw.csv")
        
    raw_df = pd.read_csv(raw_csv_path)
    print(f"Loaded {len(raw_df):,} raw measurement rows from {os.path.basename(raw_csv_path)}.")
    
    # Clean datetime
    raw_df['timestamp'] = pd.to_datetime(raw_df['collected_at']).dt.floor('h')
    
    # Remove duplicates and non-physical values
    raw_df = raw_df.drop_duplicates(subset=['station_id', 'timestamp'])
    raw_df.loc[raw_df['value'] <= 0, 'value'] = np.nan
    raw_df.loc[raw_df['value'] > 999, 'value'] = 999.0
    
    # 2. Pivot to complete hourly time grid (2023 to 2026 present)
    end_date = "2026-10-28 23:00:00" if "2026" in raw_csv_path else "2024-12-31 23:00:00"
    full_hourly_index = pd.date_range(start="2023-01-01 00:00:00", end=end_date, freq="h")
    total_hours = len(full_hourly_index)
    print(f"Complete hourly timeline: {full_hourly_index[0]} to {full_hourly_index[-1]} ({total_hours} hours)")
    
    pivot_df = raw_df.pivot(index='timestamp', columns='station_id', values='value')
    # Reindex to full timeline and enforce exact station ordering
    pivot_df = pivot_df.reindex(index=full_hourly_index, columns=station_ids)
    
    # Missingness audit before imputation
    missing_before = pivot_df.isna().sum()
    print("\nMissingness Rate per Station (Before Imputation):")
    for st in station_ids:
        st_name = stations_df.loc[stations_df['station_id'] == st, 'name'].values[0]
        pct = (missing_before[st] / total_hours) * 100
        print(f"  - {st_name:22s} ({st}): {missing_before[st]:4d} missing ({pct:.1f}%)")
        
    # 3. Controlled Imputation
    # Short gaps (<= 4h): Linear interpolation
    pivot_df = pivot_df.interpolate(method='time', limit=4)
    # Forward-fill & backward-fill remaining small residual gaps
    pivot_df = pivot_df.ffill().bfill()
    
    missing_after = pivot_df.isna().sum().sum()
    print(f"\nMissing values after controlled interpolation: {missing_after} (Dataset is continuous).")
    
    # 4. Temporal Cyclic Features (Hour of Day & Day of Week)
    hours = full_hourly_index.hour.values
    days = full_hourly_index.dayofweek.values
    
    hour_sin = np.sin(2 * np.pi * hours / 24.0)
    hour_cos = np.cos(2 * np.pi * hours / 24.0)
    day_sin = np.sin(2 * np.pi * days / 7.0)
    day_cos = np.cos(2 * np.pi * days / 7.0)
    
    # Temporal features matrix: [T, 4]
    time_features = np.stack([hour_sin, hour_cos, day_sin, day_cos], axis=1)
    
    # 5. Chronological Partitioning: 70% Train, 15% Val, 15% Test
    n_train = int(total_hours * 0.70)
    n_val = int(total_hours * 0.15)
    n_test = total_hours - n_train - n_val
    
    print(f"\nChronological Split:")
    print(f"  Train:      0 to {n_train:5d} hours ({full_hourly_index[0]} to {full_hourly_index[n_train-1]})")
    print(f"  Validation: {n_train:5d} to {n_train+n_val:5d} hours ({full_hourly_index[n_train]} to {full_hourly_index[n_train+n_val-1]})")
    print(f"  Test:       {n_train+n_val:5d} to {total_hours:5d} hours ({full_hourly_index[n_train+n_val]} to {full_hourly_index[-1]})")
    
    # 6. Feature Scaling (Fit strictly on Training period to prevent leakage)
    pm25_matrix = pivot_df.values.astype(np.float32) # [T, N_stations]
    
    train_pm25 = pm25_matrix[:n_train]
    pm25_mean = float(np.mean(train_pm25))
    pm25_std = float(np.std(train_pm25))
    print(f"\nTraining PM2.5 Statistics: Mean = {pm25_mean:.2f} µg/m³, Std = {pm25_std:.2f} µg/m³")
    
    scaler_params = {
        "mean": pm25_mean,
        "std": pm25_std,
        "input_window": INPUT_WINDOW,
        "horizons": HORIZONS,
        "station_ids": station_ids,
        "station_names": stations_df['name'].tolist()
    }
    with open(os.path.join(PROCESSED_DIR, "scaler_params.json"), "w") as f:
        json.dump(scaler_params, f, indent=2)
        
    # Standardize PM2.5 values
    norm_pm25 = (pm25_matrix - pm25_mean) / pm25_std
    
    # Combine PM2.5 with temporal features for each station
    # Station features shape at each step: [T, N_stations, 5] (PM2.5, hour_sin, hour_cos, day_sin, day_cos)
    features_all = np.zeros((total_hours, n_stations, 5), dtype=np.float32)
    for i in range(n_stations):
        features_all[:, i, 0] = norm_pm25[:, i]
        features_all[:, i, 1:] = time_features
        
    # 7. Generate Sliding Windows: Input [24, N, 5], Target [4, N] in original or standardized units
    # Note: Target Y is standardized for training stability; we evaluate in original units using scaler.
    def create_sequences(start_idx, end_idx):
        X_list, Y_norm_list, Y_orig_list = [], [], []
        max_horizon = max(HORIZONS)
        for t in range(start_idx + INPUT_WINDOW, end_idx - max_horizon):
            # Input window: [t - INPUT_WINDOW : t] -> shape: [24, N, 5]
            x_window = features_all[t - INPUT_WINDOW : t]
            
            # Future horizons: t + 1, t + 3, t + 6, t + 12 -> shape: [4, N]
            y_norm = np.stack([norm_pm25[t + h] for h in HORIZONS], axis=0)
            y_orig = np.stack([pm25_matrix[t + h] for h in HORIZONS], axis=0)
            
            X_list.append(x_window)
            Y_norm_list.append(y_norm)
            Y_orig_list.append(y_orig)
            
        return np.array(X_list, dtype=np.float32), np.array(Y_norm_list, dtype=np.float32), np.array(Y_orig_list, dtype=np.float32)

    print("\nGenerating Sliding Windows...")
    X_train, Y_train_norm, Y_train_orig = create_sequences(0, n_train)
    X_val, Y_val_norm, Y_val_orig = create_sequences(n_train - INPUT_WINDOW, n_train + n_val)
    X_test, Y_test_norm, Y_test_orig = create_sequences(n_train + n_val - INPUT_WINDOW, total_hours)
    
    print(f"  Training Samples:   X={X_train.shape}, Y={Y_train_norm.shape}")
    print(f"  Validation Samples: X={X_val.shape}, Y={Y_val_norm.shape}")
    print(f"  Testing Samples:    X={X_test.shape}, Y={Y_test_norm.shape}")
    
    # 8. Save compressed datasets
    np.savez_compressed(
        os.path.join(PROCESSED_DIR, "train_dataset.npz"),
        X=X_train, Y_norm=Y_train_norm, Y_orig=Y_train_orig
    )
    np.savez_compressed(
        os.path.join(PROCESSED_DIR, "val_dataset.npz"),
        X=X_val, Y_norm=Y_val_norm, Y_orig=Y_val_orig
    )
    np.savez_compressed(
        os.path.join(PROCESSED_DIR, "test_dataset.npz"),
        X=X_test, Y_norm=Y_test_norm, Y_orig=Y_test_orig
    )
    print("Saved train_dataset.npz, val_dataset.npz, and test_dataset.npz to data/processed/")
    
    # 9. Plot Raw vs Preprocessed PM2.5 Time Series for a sample station
    plt.figure(figsize=(12, 4.5))
    sample_station = "site_301" # Anand Vihar
    st_idx = station_ids.index(sample_station)
    st_name = stations_df.loc[stations_df['station_id'] == sample_station, 'name'].values[0]
    
    plt.plot(full_hourly_index, pm25_matrix[:, st_idx], color='#2563EB', linewidth=0.8, alpha=0.85, label=f'{st_name} (Hourly PM2.5)')
    plt.axvline(full_hourly_index[n_train], color='#10B981', linestyle='--', linewidth=1.5, label='Validation Split')
    plt.axvline(full_hourly_index[n_train + n_val], color='#EF4444', linestyle='--', linewidth=1.5, label='Test Split')
    
    year_label = "2023-2026" if "2026" in raw_csv_path else "2023-2024"
    plt.title(f"{year_label} PM2.5 Concentration Series & Chronological Splits ({st_name})", fontsize=11, fontweight='bold')
    plt.ylabel("PM2.5 (µg/m³)", fontsize=10)
    plt.xlabel("Date", fontsize=10)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    split_plot_path = os.path.join(PLOTS_DIR, "chronological_split_series.png")
    plt.savefig(split_plot_path, dpi=300)
    plt.close()
    print(f"Saved time-series split visualization to: {split_plot_path}")
    print("=" * 65)

if __name__ == "__main__":
    preprocess_air_quality_data()
