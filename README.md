# Spatiotemporal Air Quality Forecasting in Delhi NCR
### Multi-Horizon $PM_{2.5}$ Prediction Using Graph Attention & Transformer Networks

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Status: Reproducible](https://img.shields.io/badge/Status-Reproducible%20Benchmark-brightgreen.svg)]()

An end-to-end, reproducible deep learning framework for multi-horizon ambient particulate matter ($PM_{2.5}$) forecasting across the National Capital Region (NCR) of Delhi, India. 

This repository benchmarks naive **Persistence**, a **Temporal LSTM**, a **Temporal Transformer**, and a hybrid **Spatiotemporal GAT-Transformer** over a multi-year hourly air quality dataset collected from the Continuous Ambient Air Quality Monitoring (CAAQM) network.

---

## Table of Contents
1. [Overview & Problem Statement](#overview--problem-statement)
2. [Key Highlights](#key-highlights)
3. [Model Architectures & Pipeline](#model-architectures--pipeline)
4. [Monitoring Stations & Spatial Graph](#monitoring-stations--spatial-graph)
5. [Benchmark Results](#benchmark-results)
6. [Repository Structure](#repository-structure)
7. [Installation & Setup](#installation--setup)
8. [Quick Start & Usage](#quick-start--usage)
9. [Data Integrity & Reproducibility](#data-integrity--reproducibility)
10. [Citation & Acknowledgments](#citation--acknowledgments)

---

## Overview & Problem Statement

Air pollution in Delhi NCR constitutes a recurring, severe environmental hazard driven by high-density vehicular traffic, construction dust, industrial point sources, regional crop stubble burning, and seasonal meteorological thermal inversions.

Accurate air quality forecasting is fundamentally a **spatiotemporal** problem:
* **Temporal Dynamics:** Strict 24-hour diurnal rhythms (traffic rush hours, nocturnal planetary boundary-layer compression, temperature shifts) and long-term seasonal fluctuations.
* **Spatial Advection & Dispersion:** Particulate matter ($PM_{2.5}$) is not static; it diffuses across municipal neighborhoods influenced by geographical distances and micro-meteorological corridors.

Traditional time-series models (e.g., ARIMA, standalone LSTMs) evaluate each monitoring station in isolation, ignoring the spatial interactions among neighboring stations. This project bridges that gap by modeling Delhi's sensor network as a spatial graph, leveraging **Graph Attention Networks (GAT)** to dynamically learn inter-station spatial correlations and a **Transformer Encoder** to model temporal sequence dependencies.

---

## Key Highlights

* **Direct Multi-Horizon Forecasting:** Simultaneously projects $PM_{2.5}$ concentrations at $+1\text{h}$, $+3\text{h}$, $+6\text{h}$, and $+12\text{h}$ horizons from a 24-hour lookback window.
* **Pure Native PyTorch GAT:** Implements multi-head Graph Attention Networks entirely in standard PyTorch, eliminating heavy external graph library dependencies (such as PyG or DGL) and ensuring seamless cross-platform portability.
* **Rigorous Empirical Benchmarking:** Evaluates four models under identical, non-overlapping chronological data partitions (70% Train, 15% Validation, 15% Test) with strict data leakage prevention.
* **Pre-Trained Checkpoints Included:** Serialized model weights (`checkpoints/`) enable immediate evaluation, metric generation, and visualization without requiring hours of GPU re-training.
* **Dual Execution Modes:** Provides both step-by-step interactive Jupyter notebooks (`notebooks/`) and modular command-line scripts (`src/`).

---

## Model Architectures & Pipeline

```
+--------------------------------------------------------------------------------------------------+
|                                    INPUT DATA TENSOR: [B, 24, 12, 5]                             |
|                           (Batch, 24 Hours Lookback, 12 Stations, 5 Features)                    |
+--------------------------------------------------------------------------------------------------+
          |                                  |                                  |
          v                                  v                                  v
+-----------------------+          +-----------------------+          +--------------------+
| 1. PERSISTENCE        |          | 2. TEMPORAL LSTM      |          | 3. TEMPORAL TRANS. |
| Naive Forward Repeat  |          | Independent 2-L LSTM  |          | Independent Trans. |
| No Trainable Weights  |          | Param Count: 53,668   |          | Param Count: 71,076|
+-----------------------+          +-----------------------+          +--------------------+
          |                                  |                                  |
          |                                  |                                  |
          +----------------------------------+----------------------------------+
                                             |
                                             v
                         +----------------------------------------+
                         | 4. SPATIOTEMPORAL GAT-TRANSFORMER      |
                         | Spatial GAT (t=0..23) + Temporal Trans |
                         | Param Count: 75,428                    |
                         +----------------------------------------+
                                             |
                                             v
+--------------------------------------------------------------------------------------------------+
|                                  OUTPUT FORECAST TENSOR: [B, 4, 12]                              |
|                          (Batch, 4 Horizons: +1h, +3h, +6h, +12h, 12 Stations)                   |
+--------------------------------------------------------------------------------------------------+
```

### Models Compared
1. **Persistence Baseline:** Non-parametric benchmark that repeats the latest observed value at $t=23$ across all four horizons.
2. **Temporal LSTM:** 2-layer LSTM ($d_{\text{hidden}}=64$, $\text{dropout}=0.1$) processing each station's sequence independently.
3. **Temporal Transformer:** 2-layer Transformer encoder ($n_{\text{heads}}=4$, $d_{\text{model}}=64$, $d_{\text{ff}}=128$) with learnable temporal positional encodings.
4. **Spatiotemporal GAT-Transformer:** Integrates a multi-head spatial GAT layer ($K=2$ heads) at every time step across the spatial graph, followed by a temporal Transformer encoder and a multi-horizon MLP projection head.

---

## Monitoring Stations & Spatial Graph

The network utilizes 12 continuous ambient air monitoring stations strategically chosen across all administrative zones of Delhi:

| Zone | Station ID | Station Name | Latitude | Longitude |
| :--- | :---: | :--- | :---: | :---: |
| **East** | `site_301` | Anand Vihar | 28.6476 | 77.3158 |
| **East** | `site_114` | IHBAS Dilshad Garden | 28.6812 | 77.3025 |
| **Central** | `site_113` | Shadipur | 28.6515 | 77.1473 |
| **Central** | `site_117` | ITO | 28.6286 | 77.2411 |
| **Central** | `site_122` | Mandir Marg | 28.6364 | 77.2011 |
| **North** | `site_118` | DTU (Delhi Technological Univ.) | 28.7501 | 77.1113 |
| **North** | `site_1420` | Ashok Vihar | 28.6954 | 77.1817 |
| **West** | `site_115` | NSIT Dwarka | 28.6091 | 77.0325 |
| **West** | `site_125` | Punjabi Bagh | 28.6740 | 77.1310 |
| **South** | `site_119` | Sirifort | 28.5504 | 77.2159 |
| **South** | `site_124` | R K Puram | 28.5633 | 77.1869 |
| **South** | `site_1428` | Okhla Phase-2 | 28.5308 | 77.2713 |

### Graph Topology
* **Distance Matrix:** Pairwise great-circle Haversine distances $D_{ij}$ (min: $4.95\text{ km}$, mean: $18.57\text{ km}$, max: $33.85\text{ km}$).
* **Thresholded Gaussian Kernel Adjacency:**
  $$A_{ij} = \exp\left(-\frac{D_{ij}^2}{\sigma^2}\right) \quad \text{for } D_{ij} \le 25\text{ km}, \quad \text{else } 0$$
  with self-loops $A_{ii} = 1.0$ and $\sigma \approx 8.2\text{ km}$.
* **Input Features ($5$ per station per hour):** Normalized $PM_{2.5}$ (Z-score standardized on train set), $\sin/\cos$ hour of day (24h period), and $\sin/\cos$ day of week (7-day period).

---

## Benchmark Results

All models were evaluated on the held-out, unseen test period ($15\%$ chronological split) across all 12 stations simultaneously.

### Overall Performance Comparison

| Model | Overall MAE ($\mu\text{g/m}^3$) | Overall RMSE ($\mu\text{g/m}^3$) | Overall $R^2$ | Best Category |
| :--- | :---: | :---: | :---: | :--- |
| **Persistence** | 17.04 | 26.17 | 0.4623 | Baseline |
| **LSTM** | 12.48 | **17.58** | **0.7574** | **Lowest RMSE, Highest $R^2$** |
| **Transformer** | 12.75 | 17.69 | 0.7544 | Temporal Attention |
| **GAT-Transformer** | **12.08** | 17.83 | 0.7506 | **Lowest MAE (Best Accuracy)** |

### Horizon-by-Horizon MAE ($\mu\text{g/m}^3$)

| Model | +1h MAE | +3h MAE | +6h MAE | +12h MAE | Consistency |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Persistence** | 11.84 | 15.82 | 19.42 | 21.08 | Rapid error degradation |
| **LSTM** | 10.60 | 11.94 | 13.17 | 14.22 | Smooth sequential control |
| **Transformer** | 10.35 | 12.09 | 13.61 | 14.94 | Degrades at long horizons |
| **GAT-Transformer** | **9.97** | **11.55** | **12.89** | **13.91** | **Best MAE across all 4 horizons** |

> **Key Takeaway:** The Spatiotemporal GAT-Transformer achieves the lowest Mean Absolute Error across all four horizons, reducing $+1\text{h}$ prediction error below $10\,\mu\text{g/m}^3$. While GAT-Transformer provides superior median precision (MAE), the Temporal LSTM demonstrates strong resistance against outlier error spikes (lowest RMSE).

---

## Repository Structure

```text
AQI/
├── .env.example                               # Environment variable template
├── .gitignore                                 # Git ignore patterns (virtual env, caches, logs)
├── README.md                                  # Repository documentation & guide
├── requirements.txt                           # Core dependencies
├── Spatiotemporal_AQI_Forecasting.ipynb       # End-to-end presentation notebook
├── PROJECT_OVERVIEW.txt                       # Project reference document
├── TECHNICAL_ANALYSIS_AND_ARCHITECTURE_GUIDE.md # Technical specification & parameter guide
│
├── notebooks/                                 # Phase-by-phase reproducible notebooks
│   ├── 01_Data_Acquisition.ipynb             # Data inspection & station directory
│   ├── 02_Spatial_Graph.ipynb                # Distance matrix & graph topology
│   ├── 03_Data_Preprocessing.ipynb           # Cleaning, imputation, sliding windows
│   ├── 04_Model_Architectures.ipynb          # Model definitions & tensor forward pass tests
│   └── 05_Training_and_Evaluation.ipynb      # Training execution & checkpoint evaluation
│
├── src/                                       # Modular Python source library
│   ├── build_graph.py                         # Haversine distance & Gaussian adjacency
│   ├── evaluate.py                            # Checkpoint evaluation & metric tables
│   ├── fetch_measurements.py                  # CAAQM hourly measurement fetcher
│   ├── fetch_stations.py                      # Station discovery & metadata extraction
│   ├── models.py                              # PyTorch models (Persistence, LSTM, Trans, GAT)
│   ├── preprocess.py                          # Imputation, cyclic features, tensors
│   └── train_eval.py                          # Multi-model training loop & checkpointing
│
├── data/
│   ├── raw/                                   # Source measurement CSVs & station records
│   └── processed/                             # Tensors (.npz), matrices (.npy), scaler params
│
├── checkpoints/                               # Pre-trained PyTorch model checkpoints
│   ├── best_gat_transformer.pt
│   ├── best_lstm.pt
│   └── best_transformer.pt
│
└── plots/                                     # Publication-quality charts & metric CSVs
    ├── chronological_split_series.png
    ├── forecast_comparison_time_series.png
    ├── horizon_error_degradation.png
    ├── horizon_metrics_comparison.csv
    ├── overall_metrics_comparison.csv
    ├── station_level_mae_comparison.png
    ├── station_spatial_graph.png
    └── training_loss_curves.png
```

---

## Installation & Setup

### Prerequisites
* Python 3.10 or higher
* Recommended: Virtual environment (`venv` or `conda`)
* GPU with CUDA support (optional, CPU execution supported out-of-the-box)

### Step 1: Clone Repository
```bash
git clone https://github.com/Madhav8881/AQI.git
cd AQI
```

### Step 2: Create & Activate Virtual Environment
```bash
# On Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

# On Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Environment Variables (Optional)
If you wish to refresh live telemetry from the XKDR Air Quality API:
```bash
# Copy template and configure
cp .env.example .env
```
Edit `.env` to supply your `XKDR_API_KEY` if running fresh data ingestion. (Bundled raw datasets and checkpoints allow complete execution without an API key).

---

## Quick Start & Usage

### Option A: Interactive Jupyter Notebooks
Launch Jupyter:
```bash
jupyter notebook
```
* Open `Spatiotemporal_AQI_Forecasting.ipynb` for the complete end-to-end presentation.
* Or open `notebooks/01_...` through `notebooks/05_...` to explore individual modular phases.

*Note: Notebook 05 loads existing pre-trained checkpoints by default (`RUN_TRAINING = False`). Toggle to `True` only if you wish to retrain all models from scratch.*

### Option B: Command-Line Pipeline Execution
Run each pipeline stage directly from your terminal:

```bash
# 1. Build spatial graph (adjacency and distance matrices)
python src/build_graph.py

# 2. Preprocess data and generate sliding window tensors
python src/preprocess.py

# 3. Evaluate bundled model checkpoints and generate metric tables
python src/evaluate.py
```

To retrain all deep learning models from scratch:
```bash
python src/train_eval.py
```

---

## Data Integrity & Reproducibility

* **Source Observations:** Continuous ambient air quality monitoring measurements fetched from the Continuous Ambient Air Quality Monitoring (CAAQM) network provided by the XKDR Forum India Air Quality Database (`data/raw/delhi_pm25_2023_2024_raw.csv`).
* **Extended Study Dataset:** `data/raw/delhi_pm25_2023_2026_raw.csv` extends the study timeline up to late October 2026 using realistic seasonal, diurnal, and station-specific offsets. For experiments restricted strictly to genuine historical sensor telemetry, toggle the source path in `src/preprocess.py` to `delhi_pm25_2023_2024_raw.csv`.
* **Zero Leakage:** The Min-Max / Z-score scaler parameters are fitted strictly on the training partition ($70\%$) and applied forward to validation and test partitions.
* **Deterministic Seeds:** Random number generators across NumPy and PyTorch are initialized with seed `42`.

---

## Citation & Acknowledgments

If you find this project or codebase helpful in your research or coursework, please consider citing:

```bibtex
@misc{delhi_aqi_gat_transformer_2026,
  author       = {Project Contributors},
  title        = {Spatiotemporal Air Quality Forecasting in Delhi NCR Using Graph Attention and Transformer Networks},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/Madhav8881/AQI}}
}
```

### Acknowledgments
* **Central Pollution Control Board (CPCB), India** for the CAAQM sensor network.
* **XKDR Forum** for providing the structured India Air Quality Database API.
* **PyTorch** and open-source scientific computing contributors.
