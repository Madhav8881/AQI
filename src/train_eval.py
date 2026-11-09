"""
Phase 5 & 6: Training, Checkpointing, and Comprehensive Academic Evaluation.
Trains LSTM, Transformer, and GAT-Transformer; Evaluates all models against Persistence baseline.
Generates metrics table (MAE, RMSE, R2) and publication-quality plots.
"""
import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from models import PersistenceModel, TemporalLSTM, TemporalTransformer, SpatiotemporalGATTransformer

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
CHECKPOINTS_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")

os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Device configuration (CPU or GPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 20
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

def train_single_model(model_name, model, train_loader, val_loader, adj_matrix, scaler_std, scaler_mean):
    print("\n" + "-" * 60)
    print(f" Training Model: {model_name} on {DEVICE} ")
    print("-" * 60)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_path = os.path.join(CHECKPOINTS_DIR, f"best_{model_name.lower().replace('-', '_')}.pt")
    
    history = {"train_loss": [], "val_loss": []}
    adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32, device=DEVICE)
    
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        # 1. Training Phase
        model.train()
        total_train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            
            if "GAT" in model_name:
                preds = model(batch_x, adj_tensor)
            else:
                preds = model(batch_x)
                
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_train_loss += loss.item() * len(batch_x)
            
        train_loss = total_train_loss / len(train_loader.dataset)
        
        # 2. Validation Phase
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                if "GAT" in model_name:
                    preds = model(batch_x, adj_tensor)
                else:
                    preds = model(batch_x)
                loss = criterion(preds, batch_y)
                total_val_loss += loss.item() * len(batch_x)
                
        val_loss = total_val_loss / len(val_loader.dataset)
        scheduler.step(val_loss)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        # Checkpoint if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            saved_flag = " [*Best Saved]"
        else:
            saved_flag = ""
            
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}{saved_flag}")
        
    elapsed = time.time() - start_time
    print(f"Completed {model_name} in {elapsed:.1f}s. Best Val Loss: {best_val_loss:.4f}")
    
    # Reload best model weights
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    return model, history

def main():
    print("=" * 65)
    print(" Phase 5 & 6: Training & Multi-Horizon Evaluation ")
    print("=" * 65)
    
    # 1. Load Data
    train_npz = np.load(os.path.join(PROCESSED_DIR, "train_dataset.npz"))
    val_npz = np.load(os.path.join(PROCESSED_DIR, "val_dataset.npz"))
    test_npz = np.load(os.path.join(PROCESSED_DIR, "test_dataset.npz"))
    adj_matrix = np.load(os.path.join(PROCESSED_DIR, "adjacency_matrix.npy"))
    
    with open(os.path.join(PROCESSED_DIR, "scaler_params.json"), "r") as f:
        scaler = json.load(f)
    pm25_mean = scaler['mean']
    pm25_std = scaler['std']
    station_names = scaler['station_names']
    
    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_npz['X']), torch.tensor(train_npz['Y_norm'])),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(val_npz['X']), torch.tensor(val_npz['Y_norm'])),
        batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(test_npz['X']), torch.tensor(test_npz['Y_norm'])),
        batch_size=BATCH_SIZE, shuffle=False
    )
    
    # Ground truth for test set in original units (µg/m³)
    Y_test_orig = test_npz['Y_orig'] # [N_samples, 4, 12]
    X_test_tensor = torch.tensor(test_npz['X'], dtype=torch.float32).to(DEVICE)
    adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32).to(DEVICE)
    
    models = {
        "LSTM": TemporalLSTM().to(DEVICE),
        "Transformer": TemporalTransformer().to(DEVICE),
        "GAT-Transformer": SpatiotemporalGATTransformer().to(DEVICE)
    }
    
    histories = {}
    
    # 2. Train Models
    for name, model in models.items():
        trained_model, hist = train_single_model(
            name, model, train_loader, val_loader, adj_matrix, pm25_std, pm25_mean
        )
        models[name] = trained_model
        histories[name] = hist
        
    # 3. Add Persistence Model
    persistence = PersistenceModel(horizons=HORIZONS)
    
    # 4. Generate Predictions on Test Set
    print("\n" + "=" * 65)
    print(" Generating Predictions on Unseen Chronological Test Period ")
    print("=" * 65)
    
    predictions = {}
    
    # Persistence predictions
    with torch.no_grad():
        y_pers_norm = persistence.predict(X_test_tensor).cpu().numpy()
        predictions["Persistence"] = y_pers_norm * pm25_std + pm25_mean
        
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            if "GAT" in name:
                y_pred_norm = model(X_test_tensor, adj_tensor).cpu().numpy()
            else:
                y_pred_norm = model(X_test_tensor).cpu().numpy()
            predictions[name] = y_pred_norm * pm25_std + pm25_mean
            
    # 5. Evaluate Metrics: Overall & Horizon-Specific
    results = []
    horizon_results = {h: [] for h in HORIZONS}
    
    for name, pred in predictions.items():
        # Overall metrics across all horizons and stations
        mae, rmse, r2 = compute_metrics(Y_test_orig, pred)
        results.append({
            "Model": name,
            "Overall MAE (µg/m³)": round(mae, 2),
            "Overall RMSE (µg/m³)": round(rmse, 2),
            "Overall R²": round(r2, 4)
        })
        
        # Horizon-specific metrics
        for h_idx, h in enumerate(HORIZONS):
            h_mae, h_rmse, h_r2 = compute_metrics(Y_test_orig[:, h_idx, :], pred[:, h_idx, :])
            horizon_results[h].append({
                "Model": name,
                "Horizon": f"+{h}h",
                "MAE": round(h_mae, 2),
                "RMSE": round(h_rmse, 2),
                "R²": round(h_r2, 4)
            })
            
    df_results = pd.DataFrame(results)
    print("\nOVERALL TEST SET PERFORMANCE COMPARISON:")
    print(df_results.to_string(index=False))
    df_results.to_csv(os.path.join(PLOTS_DIR, "overall_metrics_comparison.csv"), index=False)
    
    # Print per-horizon breakdown
    print("\nMULTI-HORIZON ERROR BREAKDOWN:")
    horizon_rows = []
    for h in HORIZONS:
        for item in horizon_results[h]:
            horizon_rows.append(item)
    df_horizons = pd.DataFrame(horizon_rows)
    print(df_horizons.to_string(index=False))
    df_horizons.to_csv(os.path.join(PLOTS_DIR, "horizon_metrics_comparison.csv"), index=False)
    
    # 6. Generate Publication-Quality Evaluation Visualizations
    # Plot 1: Training & Validation Loss Curves
    plt.figure(figsize=(9, 5))
    colors = {"LSTM": "#3B82F6", "Transformer": "#F59E0B", "GAT-Transformer": "#10B981"}
    for name, hist in histories.items():
        plt.plot(hist['train_loss'], label=f"{name} (Train)", linestyle='--', color=colors[name], alpha=0.7)
        plt.plot(hist['val_loss'], label=f"{name} (Val)", linestyle='-', color=colors[name], linewidth=2.0)
    plt.title("Model Convergence: Training vs Validation Loss (MSE)", fontsize=11, fontweight='bold')
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Standardized MSE Loss", fontsize=10)
    plt.legend(fontsize=9, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "training_loss_curves.png"), dpi=300)
    plt.close()
    
    # Plot 2: Actual vs Predicted PM2.5 Time Series on Test Set
    # Select Anand Vihar (Station 0) at +1h and +6h horizons for a representative 14-day window
    station_idx = 0
    st_name = station_names[station_idx]
    test_slice = slice(100, 100 + 336) # 14 days (336 hours)
    
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    
    # Horizon +1h
    h1_idx = 0
    axes[0].plot(Y_test_orig[test_slice, h1_idx, station_idx], label='Ground Truth (Observed)', color='black', linewidth=1.8)
    axes[0].plot(predictions['Persistence'][test_slice, h1_idx, station_idx], label='Persistence', color='#9CA3AF', linestyle=':', linewidth=1.2)
    axes[0].plot(predictions['LSTM'][test_slice, h1_idx, station_idx], label='LSTM', color='#3B82F6', linestyle='--', linewidth=1.4)
    axes[0].plot(predictions['Transformer'][test_slice, h1_idx, station_idx], label='Transformer', color='#F59E0B', linestyle='--', linewidth=1.4)
    axes[0].plot(predictions['GAT-Transformer'][test_slice, h1_idx, station_idx], label='Proposed GAT-Transformer', color='#10B981', linewidth=2.0)
    axes[0].set_title(f"1-Hour Ahead PM2.5 Forecast: {st_name} (Delhi Winter Peak)", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("PM2.5 (µg/m³)", fontsize=10)
    axes[0].legend(loc='upper right', fontsize=8.5)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # Horizon +6h
    h6_idx = 2
    axes[1].plot(Y_test_orig[test_slice, h6_idx, station_idx], label='Ground Truth (Observed)', color='black', linewidth=1.8)
    axes[1].plot(predictions['Persistence'][test_slice, h6_idx, station_idx], label='Persistence', color='#9CA3AF', linestyle=':', linewidth=1.2)
    axes[1].plot(predictions['LSTM'][test_slice, h6_idx, station_idx], label='LSTM', color='#3B82F6', linestyle='--', linewidth=1.4)
    axes[1].plot(predictions['Transformer'][test_slice, h6_idx, station_idx], label='Transformer', color='#F59E0B', linestyle='--', linewidth=1.4)
    axes[1].plot(predictions['GAT-Transformer'][test_slice, h6_idx, station_idx], label='Proposed GAT-Transformer', color='#10B981', linewidth=2.0)
    axes[1].set_title(f"6-Hour Ahead PM2.5 Forecast: {st_name} (Delhi Winter Peak)", fontsize=11, fontweight='bold')
    axes[1].set_ylabel("PM2.5 (µg/m³)", fontsize=10)
    axes[1].set_xlabel("Time Index (Hours)", fontsize=10)
    axes[1].legend(loc='upper right', fontsize=8.5)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "forecast_comparison_time_series.png"), dpi=300)
    plt.close()
    
    # Plot 3: Multi-Horizon MAE & RMSE Degradation
    plt.figure(figsize=(10, 4.5))
    h_labels = [f"+{h}h" for h in HORIZONS]
    
    for name in predictions.keys():
        maes = [df_horizons[(df_horizons['Model'] == name) & (df_horizons['Horizon'] == h_lab)]['MAE'].values[0] for h_lab in h_labels]
        plt.plot(h_labels, maes, marker='o', label=name, linewidth=2.0)
        
    plt.title("Forecasting Error Degradation across Future Horizons (+1h to +12h)", fontsize=11, fontweight='bold')
    plt.xlabel("Forecast Horizon", fontsize=10)
    plt.ylabel("Mean Absolute Error (µg/m³)", fontsize=10)
    plt.legend(fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "horizon_error_degradation.png"), dpi=300)
    plt.close()
    
    print("\nSaved all evaluation plots and CSV tables to 'plots/' directory.")
    print("=" * 65)

if __name__ == "__main__":
    main()
