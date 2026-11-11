# Delhi PM2.5 Forecaster
### Spatiotemporal air-quality forecasting with Graph Attention + Transformer networks

Forecasts hourly PM2.5 at **36 monitoring stations across Delhi** for the next **1, 3, 6, 12 and 24 hours**,
trained on **real** measurements (Jan 2023 – Aug 2025), with an interactive **Streamlit app** where you
can pick any station, any moment, and watch the models forecast live — including what happens when a
station's sensor fails.

---

## Highlights

- **Real data only.** 721,402 hourly readings from CPCB, DPCC and IMD stations (XKDR Air Quality API). No synthetic or interpolated values are used for training targets or scoring.
- **Graph neural network.** Stations form a graph (8 nearest neighbours, distance-weighted). A Graph Attention Network learns, hour by hour, which neighbouring stations matter.
- **Leakage-free evaluation.** Chronological split, scalers fitted on training data only, causal (past-only) gap filling, metrics computed only on real observations, 3 random seeds per model, Diebold–Mariano significance tests.
- **Robust to sensor failure.** The *Robust GAT-Transformer* forecasts a station whose sensor is down with **43% lower error** than an equally trained LSTM, by learning from its neighbours.
- **Interactive demo.** Live map, per-station forecasts, backtests, model comparison, station explorer, and a one-click "simulate sensor failure" switch.

---

## Results

Test period: **Oct – Dec 2024** (winter smog) and **Apr – Aug 2025** (summer / monsoon), all 36 stations,
real observations only. Neural models: mean ± std over 3 seeds. MAE/RMSE in µg/m³.

| Model | MAE ↓ | RMSE ↓ | R² ↑ | AQI-category accuracy ↑ |
|---|---|---|---|---|
| Persistence (last value) | 36.02 | 66.94 | 0.601 | 52.8% |
| Seasonal naive (same hour yesterday) | 39.75 | 73.05 | 0.525 | 48.6% |
| LSTM | 28.36 ± 0.14 | 54.09 ± 0.26 | 0.740 | 57.6% |
| Transformer | 27.99 ± 0.34 | **52.69 ± 0.53** | **0.753** | 58.0% |
| GAT-Transformer | 27.63 ± 0.17 | 53.43 ± 0.56 | 0.746 | **58.6%** |
| Robust LSTM | 28.36 ± 0.06 | 53.97 ± 0.46 | 0.741 | 57.7% |
| **Robust GAT-Transformer** | **27.47 ± 0.27** | 52.77 ± 0.36 | 0.752 | 58.5% |

**MAE by lead time**

| Model | +1h | +3h | +6h | +12h | +24h |
|---|---|---|---|---|---|
| Persistence | 15.54 | 30.37 | 43.01 | 51.50 | 39.71 |
| LSTM | 14.49 | 25.12 | 31.03 | 34.87 | 36.32 |
| Transformer | 14.43 | 24.60 | 30.42 | 34.14 | 36.43 |
| GAT-Transformer | **14.01** | 23.71 | 29.74 | 33.95 | 36.77 |
| Robust GAT-Transformer | 14.08 | **23.64** | **29.64** | **33.77** | **36.29** |

**MAE by season / episode**

| Model | Winter smog (Oct–Dec 2024) | Summer/Monsoon (Apr–Aug 2025) | Severe episodes (> 250 µg/m³) |
|---|---|---|---|
| Persistence | 59.3 | 21.4 | 111.3 |
| LSTM | 46.8 | 16.8 | 100.9 |
| Transformer | 45.7 | 16.9 | **96.9** |
| GAT-Transformer | 45.3 | **16.6** | 100.7 |
| Robust GAT-Transformer | **44.9** | **16.6** | 98.1 |

**Significance.** The GAT-Transformer beats the LSTM and the Transformer at +1h and +3h with
p < 0.001 (Diebold–Mariano), and the LSTM at +6h (p = 0.004). At +12h and +24h the neural models are
statistically tied.

### Sensor-failure test

Each station's own last 24 h of readings is hidden in turn; the model must forecast that station anyway.

| Model | MAE, sensor working | MAE, sensor failed | Error increase |
|---|---|---|---|
| Station long-term average | — | 74.6 | — |
| LSTM | 28.2 | 55.4 | +96% |
| Transformer | 27.8 | 58.0 | +109% |
| GAT-Transformer | 27.8 | 54.5 | +96% |
| Robust LSTM | 28.4 | 42.4 | +49% |
| **Robust GAT-Transformer** | **27.4** | **31.5** | **+15%** |

"Robust" models are trained with *station dropout*: about 15% of stations have their whole input window
hidden in each training sample. The LSTM can only fall back on time-of-day / season patterns, while the
GAT learns to rebuild the missing station from its neighbours — **+1h error 24.3 vs 42.9 µg/m³**.
Station dropout also slightly *improves* normal accuracy, making the Robust GAT-Transformer the best model overall.

### Honest limitations
- **Winter is harder than summer** (MAE ≈ 45 vs 17). The test winter, Nov 2024, was more extreme than anything in training (city-average peak 838 vs 570 µg/m³ in Nov 2023), and only one full Oct–Dec season is in the training data.
- **+24h forecasts** are only modestly better than persistence; without weather inputs, day-ahead PM2.5 is fundamentally hard.
- **Jan – Mar 2025 is missing** for every station in the source data; those dates are skipped.

---

## Data

| | |
|---|---|
| Source | [XKDR Air Quality API](https://airquality.xkdr.org) (CPCB CAAQM network: CPCB, DPCC, IMD stations) |
| Stations | 36 Delhi stations with ≥ 70% real-data coverage (Chandni Chowk dropped at 38%) |
| Period | 1 Jan 2023 – 31 Aug 2025, hourly (23,376 hours) |
| Coverage | 85.7% of station-hours have a real reading |
| Cleaning | Non-physical values (≤ 0 or > 1000) and stuck sensors (≥ 12 identical consecutive hours, 1,458 readings) removed |

| Split | Period | Forecast windows |
|---|---|---|
| Train | Jan 2023 – Jun 2024 | 12,905 |
| Validation | Jul – Sep 2024 | 2,184 |
| Test | Oct – Dec 2024, Apr – Aug 2025 | 5,843 |

The split is **by time**: every station appears in every split, and the models are always tested on
months they never saw.

## Method

**Inputs per station per hour (8 features):** standardised PM2.5, an `observed` flag (1 = real reading,
0 = filled), and sine/cosine of hour-of-day, day-of-week and day-of-year. 24-hour lookback.

**No leakage:** per-station mean/std fitted on training data only; input gaps filled by forward-fill
(≤ 24 h) then station mean, never using future values; loss and metrics ignore unobserved targets.

**Graph:** Haversine distances between stations; each station links to its 8 nearest neighbours with
weight `exp(-(d/σ)²)`, σ = std of pairwise distances (8.1 km). The GAT adds `log(weight)` to its attention
scores as a distance prior that it can learn to override.

**Models** (all neural models share station embeddings and residual forecasting, so the comparison is fair):

| Model | Idea | Parameters |
|---|---|---|
| Persistence | Last observed value | 0 |
| Seasonal naive | Same hour, previous day | 0 |
| LSTM | 2-layer LSTM per station | 72k |
| Transformer | 2-layer Transformer encoder per station | 74k |
| **GAT-Transformer** | Graph attention across stations at every hour → temporal Transformer → graph attention on the summary → multi-horizon head | 91k |
| Robust variants | Same architectures + station dropout (p = 0.15) during training | — |

**Training:** AdamW (lr 1e-3, wd 1e-4), masked Huber loss, ReduceLROnPlateau, early stopping on
validation (patience 4, max 20 epochs), each epoch on a random 50% of training windows, 3 seeds per model.
Trained on CPU.

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

streamlit run app/streamlit_app.py
```

The processed dataset and trained checkpoints are included, so the app runs immediately.

### Re-running the pipeline

```bash
cd src
python fetch_measurements.py      # download real data (needs XKDR_API_KEY in .env, see .env.example)
python preprocess.py              # clean, split, build features
python build_graph.py             # station graph
python train_eval.py              # train all models x 3 seeds, then evaluate   (~4-5 h on CPU)
python train_eval.py --eval-only  # re-evaluate existing checkpoints
python sensor_failure_eval.py     # sensor-failure test
```

---

## Demo script (5 minutes)

1. **Live map** – open the app. The map shows all 36 stations coloured by AQI category on 18 Nov 2024, 8 AM (worst smog of the season). Click a station, e.g. *Anand Vihar*.
2. **Forecast** – the chart shows the last 48 h, each model's forecast for the next 24 h, and the dotted line of what actually happened. Point out the badge: this date is in the **test period**, never seen in training.
3. **Neighbour attention** – scroll down: the bar chart shows which nearby stations the GAT used for this forecast.
4. **Sensor failure** – switch on *Simulate sensor failure*. The LSTM loses its input and drifts towards an average; the **Robust GAT-Transformer** stays close to reality using the neighbours.
5. **Backtest** – show a whole month of forecasts vs reality for the selected station.
6. **Model comparison** – the results tables, significance tests, the sensor-failure chart and the per-station error map.
7. **Station explorer** – monthly pattern (winter peak) and daily cycle for any station.

---

## Repository structure

```text
AQI/
├── app/
│   └── streamlit_app.py            # interactive demo
├── src/
│   ├── config.py                   # paths, dates, horizons, AQI bands
│   ├── fetch_stations.py           # station directory (XKDR API)
│   ├── fetch_measurements.py       # real hourly PM2.5 download
│   ├── preprocess.py               # cleaning, split, features, forecast windows
│   ├── build_graph.py              # station graph
│   ├── data.py                     # dataset loading / window slicing
│   ├── models.py                   # baselines, LSTM, Transformer, GAT-Transformer
│   ├── train_eval.py               # multi-seed training
│   ├── evaluate.py                 # test metrics, significance tests
│   └── sensor_failure_eval.py      # sensor-failure test
├── data/
│   ├── raw/                        # station metadata + raw measurements
│   └── processed/                  # dataset.npz, graph matrices, scaler, station list
├── checkpoints/                    # trained models (<model>_seed<k>.pt)
├── results/                        # metrics CSVs, test predictions, training logs
├── Spatiotemporal_AQI_Forecasting.ipynb   # walkthrough notebook
└── requirements.txt
```

## Acknowledgements
Central Pollution Control Board (CPCB), Delhi Pollution Control Committee (DPCC) and India
Meteorological Department (IMD) for the monitoring network; XKDR Forum for the Air Quality API.
