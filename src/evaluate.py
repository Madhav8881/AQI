"""
Phase 6: Multi-Horizon Model Evaluation and Visualization Script.
Loads trained checkpoints (best_lstm.pt, best_transformer.pt, best_gat_transformer.pt),
generates test predictions across +1h, +3h, +6h, +12h, calculates MAE/RMSE/R2,
and outputs publication-quality evaluation figures.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from models import PersistenceModel, TemporalLSTM, TemporalTransformer, SpatiotemporalGATTransformer

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
CHECKPOINTS_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HORIZONS = [1, 3, 6, 12]

def compute_metrics(y_true, y_pred):
    """Calculate MAE, RMSE, and R2 score between ground truth and predictions."""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    # R2 Score: 1 - SS_res / SS_tot
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))
    return mae, rmse, r2

def main():
    print("=" * 65)
    print(" Phase 6: Multi-Horizon Model Evaluation on Unseen Test Set ")
    print("=" * 65)
    
    # 1. Load Data
    test_npz = np.load(os.path.join(PROCESSED_DIR, "test_dataset.npz"))
    adj_matrix = np.load(os.path.join(PROCESSED_DIR, "adjacency_matrix.npy"))
    
    with open(os.path.join(PROCESSED_DIR, "scaler_params.json"), "r") as f:
        scaler = json.load(f)
    pm25_mean = scaler['mean']
    pm25_std = scaler['std']
    station_names = scaler['station_names']
    
    X_test = test_npz['X']
    Y_test_orig = test_npz['Y_orig'] # [N_samples, 4, 12] in µg/m³
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32).to(DEVICE)
    
    # 2. Load Checkpointed Models
    models = {
        "Persistence": PersistenceModel(horizons=HORIZONS),
        "LSTM": TemporalLSTM().to(DEVICE),
        "Transformer": TemporalTransformer().to(DEVICE),
        "GAT-Transformer": SpatiotemporalGATTransformer().to(DEVICE)
    }
    
    print("\nLoading Trained Model Weights:")
    for name in ["LSTM", "Transformer", "GAT-Transformer"]:
        chk_name = f"best_{name.lower().replace('-', '_')}.pt"
        chk_path = os.path.join(CHECKPOINTS_DIR, chk_name)
        models[name].load_state_dict(torch.load(chk_path, map_location=DEVICE, weights_only=True))
        models[name].eval()
        print(f"  --> Loaded {name} from checkpoints/{chk_name}")
        
    # 3. Generate Predictions (inverted back to original µg/m³)
    print("\nGenerating Test Predictions across 2,621 sliding windows...")
    predictions = {}
    
    with torch.no_grad():
        # Persistence
        y_pers_norm = models["Persistence"].predict(X_test_tensor).cpu().numpy()
        predictions["Persistence"] = y_pers_norm * pm25_std + pm25_mean
        
        # Deep learning models
        for name in ["LSTM", "Transformer", "GAT-Transformer"]:
            if "GAT" in name:
                y_pred_norm = models[name](X_test_tensor, adj_tensor).cpu().numpy()
            else:
                y_pred_norm = models[name](X_test_tensor).cpu().numpy()
            predictions[name] = y_pred_norm * pm25_std + pm25_mean
            
    # 4. Compute Metrics: Overall & Horizon-Specific
    overall_results = []
    horizon_results = []
    
    for name, pred in predictions.items():
        # Overall metrics
        mae, rmse, r2 = compute_metrics(Y_test_orig, pred)
        overall_results.append({
            "Model": name,
            "Overall MAE (µg/m³)": round(mae, 2),
            "Overall RMSE (µg/m³)": round(rmse, 2),
            "Overall R²": round(r2, 4)
        })
        
        # Horizon metrics (+1h, +3h, +6h, +12h)
        for h_idx, h in enumerate(HORIZONS):
            h_mae, h_rmse, h_r2 = compute_metrics(Y_test_orig[:, h_idx, :], pred[:, h_idx, :])
            horizon_results.append({
                "Model": name,
                "Horizon": f"+{h}h",
                "MAE (µg/m³)": round(h_mae, 2),
                "RMSE (µg/m³)": round(h_rmse, 2),
                "R²": round(h_r2, 4)
            })
            
    df_overall = pd.DataFrame(overall_results)
    df_horizons = pd.DataFrame(horizon_results)
    
    print("\n" + "=" * 65)
    print(" OVERALL TEST SET PERFORMANCE (12 Delhi Stations, Sep-Dec 2024)")
    print("=" * 65)
    print(df_overall.to_string(index=False))
    df_overall.to_csv(os.path.join(PLOTS_DIR, "overall_metrics_comparison.csv"), index=False)
    
    print("\n" + "=" * 65)
    print(" MULTI-HORIZON ERROR BREAKDOWN (+1h, +3h, +6h, +12h)")
    print("=" * 65)
    print(df_horizons.to_string(index=False))
    df_horizons.to_csv(os.path.join(PLOTS_DIR, "horizon_metrics_comparison.csv"), index=False)
    
    # 5. Generate Visualizations
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Plot 1: Actual vs Predicted PM2.5 Time Series on Test Set
    # Representative 10-day period during winter smog peak
    station_idx = 0 # Anand Vihar
    st_name = station_names[station_idx]
    test_slice = slice(150, 150 + 240) # 10 days = 240 hours
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Horizon +1h
    h1_idx = 0
    axes[0].plot(Y_test_orig[test_slice, h1_idx, station_idx], label='Ground Truth (Actual)', color='#111827', linewidth=2.0)
    axes[0].plot(predictions['Persistence'][test_slice, h1_idx, station_idx], label='Persistence', color='#9CA3AF', linestyle=':', linewidth=1.2)
    axes[0].plot(predictions['LSTM'][test_slice, h1_idx, station_idx], label='LSTM Baseline', color='#3B82F6', linestyle='--', linewidth=1.5)
    axes[0].plot(predictions['Transformer'][test_slice, h1_idx, station_idx], label='Transformer Baseline', color='#F59E0B', linestyle='--', linewidth=1.5)
    axes[0].plot(predictions['GAT-Transformer'][test_slice, h1_idx, station_idx], label='Proposed GAT-Transformer', color='#10B981', linewidth=2.2)
    axes[0].set_title(f"1-Hour Ahead PM2.5 Forecast: {st_name} (Delhi Winter Smog Peak)", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("PM2.5 Concentration (µg/m³)", fontsize=10)
    axes[0].legend(loc='upper right', fontsize=8.5)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # Horizon +6h
    h6_idx = 2
    axes[1].plot(Y_test_orig[test_slice, h6_idx, station_idx], label='Ground Truth (Actual)', color='#111827', linewidth=2.0)
    axes[1].plot(predictions['Persistence'][test_slice, h6_idx, station_idx], label='Persistence', color='#9CA3AF', linestyle=':', linewidth=1.2)
    axes[1].plot(predictions['LSTM'][test_slice, h6_idx, station_idx], label='LSTM Baseline', color='#3B82F6', linestyle='--', linewidth=1.5)
    axes[1].plot(predictions['Transformer'][test_slice, h6_idx, station_idx], label='Transformer Baseline', color='#F59E0B', linestyle='--', linewidth=1.5)
    axes[1].plot(predictions['GAT-Transformer'][test_slice, h6_idx, station_idx], label='Proposed GAT-Transformer', color='#10B981', linewidth=2.2)
    axes[1].set_title(f"6-Hour Ahead PM2.5 Forecast: {st_name} (Delhi Winter Smog Peak)", fontsize=11, fontweight='bold')
    axes[1].set_ylabel("PM2.5 Concentration (µg/m³)", fontsize=10)
    axes[1].set_xlabel("Time Step (Hours into Test Period)", fontsize=10)
    axes[1].legend(loc='upper right', fontsize=8.5)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    pred_plot_path = os.path.join(PLOTS_DIR, "forecast_comparison_time_series.png")
    plt.savefig(pred_plot_path, dpi=300)
    plt.close()
    print(f"\nSaved time series forecast comparison to: {pred_plot_path}")
    
    # Plot 2: Horizon Degradation Chart (MAE and RMSE vs Horizon)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    h_labels = [f"+{h}h" for h in HORIZONS]
    model_colors = {"Persistence": "#6B7280", "LSTM": "#3B82F6", "Transformer": "#F59E0B", "GAT-Transformer": "#10B981"}
    
    for name in predictions.keys():
        sub = df_horizons[df_horizons['Model'] == name]
        axes[0].plot(h_labels, sub['MAE (µg/m³)'], marker='o', label=name, color=model_colors[name], linewidth=2.0)
        axes[1].plot(h_labels, sub['RMSE (µg/m³)'], marker='s', label=name, color=model_colors[name], linewidth=2.0)
        
    axes[0].set_title("MAE Degradation across Future Horizons", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Forecast Horizon", fontsize=10)
    axes[0].set_ylabel("Mean Absolute Error (µg/m³)", fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    axes[1].set_title("RMSE Degradation across Future Horizons", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Forecast Horizon", fontsize=10)
    axes[1].set_ylabel("Root Mean Squared Error (µg/m³)", fontsize=10)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    horizon_plot_path = os.path.join(PLOTS_DIR, "horizon_error_degradation.png")
    plt.savefig(horizon_plot_path, dpi=300)
    plt.close()
    print(f"Saved horizon degradation curves to: {horizon_plot_path}")
    
    # Plot 3: Station-Level Performance Comparison at +3h Horizon
    plt.figure(figsize=(12, 5))
    h3_idx = 1
    station_maes_gat = [float(np.mean(np.abs(Y_test_orig[:, h3_idx, s] - predictions['GAT-Transformer'][:, h3_idx, s]))) for s in range(len(station_names))]
    station_maes_lstm = [float(np.mean(np.abs(Y_test_orig[:, h3_idx, s] - predictions['LSTM'][:, h3_idx, s]))) for s in range(len(station_names))]
    
    x_pos = np.arange(len(station_names))
    width = 0.35
    
    plt.bar(x_pos - width/2, station_maes_lstm, width, label='LSTM Baseline', color='#93C5FD')
    plt.bar(x_pos + width/2, station_maes_gat, width, label='Proposed GAT-Transformer', color='#10B981')
    
    plt.title("+3h Forecast MAE across all 12 Delhi Monitoring Stations", fontsize=11, fontweight='bold')
    plt.ylabel("MAE (µg/m³)", fontsize=10)
    plt.xticks(x_pos, station_names, rotation=35, ha='right', fontsize=9)
    plt.legend(fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.4, axis='y')
    plt.tight_layout()
    
    st_plot_path = os.path.join(PLOTS_DIR, "station_level_mae_comparison.png")
    plt.savefig(st_plot_path, dpi=300)
    plt.close()
    print(f"Saved station-level comparison to: {st_plot_path}")
    print("=" * 65)

if __name__ == "__main__":
    main()
