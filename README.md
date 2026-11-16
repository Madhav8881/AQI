# Delhi PM2.5 Forecaster
### Spatiotemporal air-quality forecasting with Graph Attention + Transformer networks

Forecasts hourly PM2.5 at **36 monitoring stations across Delhi** for the next **1, 3, 6, 12 and 24 hours**.
Trained on real measurements from Jan 2023 â€“ Jun 2024 and tested on **two full unseen years
(Oct 2024 â€“ Sep 2026)**. Comes with an interactive **Streamlit app** that shows **live PM2.5 for Delhi
right now**, forecasts the next 24 hours, and scores those live forecasts as the real readings arrive.

---

## Highlights

- **Real data only.** Hourly PM2.5 from 36 CPCB / DPCC / IMD stations: XKDR Air Quality API (2023 â€“ Aug 2025) and the CPCB hourly files (2025 â€“ Sep 2026). The two sources agree exactly on all 117,055 overlapping readings. No synthetic values.
- **Graph neural network.** Stations form a graph (8 nearest neighbours, distance-weighted). A Graph Attention Network learns, hour by hour, which neighbouring stations matter.
- **Rigorous evaluation.** Chronological split, scalers fitted on training data only, causal (past-only) gap filling, metrics on real observations only, 3 random seeds per model, Dieboldâ€“Mariano significance tests, and a test set that runs **more than two years past the training data**.
- **Robust to sensor failure.** The *Robust GAT-Transformer* forecasts a station whose sensor is down with **only +17% error** (vs +48% for an equally trained LSTM), by learning from its neighbours.
- **Live mode, no maintenance.** Final models trained on all data up to Sep 2026 and fine-tuned once to forecast from just the current live readings. Open the app any time: it fetches live PM2.5, forecasts the next 24 hours, saves the forecast and scores it when the real readings arrive. No daily retraining.

---

## Results

Test period: **Oct 2024 â€“ Sep 2026** (17,393 hourly forecast origins Ã— 36 stations), never seen in
training. Real observations only. Neural models: mean Â± std over 3 seeds. MAE/RMSE in Âµg/mÂ³.

| Model | MAE â†“ | RMSE â†“ | RÂ² â†‘ | AQI-category accuracy â†‘ |
|---|---|---|---|---|
| Persistence (last value) | 36.06 | 63.33 | 0.566 | 51.2% |
| Seasonal naive (same hour yesterday) | 38.77 | 66.76 | 0.518 | 47.8% |
| LSTM | 27.65 Â± 0.01 | 48.77 Â± 0.14 | 0.743 | 56.8% |
| Transformer | 27.35 Â± 0.22 | 47.82 Â± 0.23 | **0.753** | 57.2% |
| **GAT-Transformer** | **26.94 Â± 0.10** | 47.95 Â± 0.33 | 0.751 | **57.8%** |
| Robust LSTM | 27.61 Â± 0.16 | 48.60 Â± 0.30 | 0.744 | 56.9% |
| Robust GAT-Transformer | 26.96 Â± 0.29 | **47.81 Â± 0.37** | **0.753** | 57.6% |

**MAE by lead time**

| Model | +1h | +3h | +6h | +12h | +24h |
|---|---|---|---|---|---|
| Persistence | 16.02 | 30.91 | 43.32 | 51.33 | 38.74 |
| LSTM | 14.74 | 24.80 | 30.27 | 33.69 | 34.79 |
| Transformer | 14.72 | 24.43 | 29.88 | 33.06 | 34.66 |
| **GAT-Transformer** | **14.34** | **23.65** | 29.25 | **32.85** | 34.63 |
| Robust GAT-Transformer | 14.43 | 23.68 | **29.23** | 32.98 | **34.54** |

**MAE by period / season**

| Model | Test year 1 (Oct 24â€“Sep 25) | Test year 2 (Oct 25â€“Sep 26) | **2026 only** | Winter (Octâ€“Feb) | Summer/Monsoon (Aprâ€“Sep) | Severe (> 250) |
|---|---|---|---|---|---|---|
| Persistence | 35.47 | 36.67 | 29.10 | 54.40 | 20.90 | 106.97 |
| LSTM | 27.56 | 27.74 | 22.46 | 41.81 | 16.56 | 93.30 |
| Transformer | 27.25 | 27.45 | 22.27 | 41.06 | 16.57 | **90.30** |
| **GAT-Transformer** | 26.86 | **27.02** | **21.96** | 40.53 | **16.28** | 92.13 |
| Robust GAT-Transformer | **26.76** | 27.18 | 22.04 | **40.49** | 16.33 | 91.40 |

- **About 25% lower error than persistence**, and **no decline in accuracy in the second test year**: in 2026 the model is still as good as right after training.
- **Significance (Dieboldâ€“Mariano):** the GAT-Transformer beats the LSTM at +1h, +3h, +6h and +12h, and the Transformer at +1h, +3h and +6h (all p < 0.001). At +24h the neural models are statistically tied.

### Sensor-failure test

Each station's own last 24 h of readings is hidden in turn; the model must forecast that station anyway.

| Model | MAE, sensor working | MAE, sensor failed | Error increase |
|---|---|---|---|
| Station long-term average | â€” | 69.8 | â€” |
| LSTM | 27.6 | 52.6 | +90% |
| Transformer | 27.3 | 54.2 | +98% |
| GAT-Transformer | 26.9 | 51.3 | +91% |
| Robust LSTM | 27.6 | 40.9 | +48% |
| **Robust GAT-Transformer** | **26.8** | **31.4** | **+17%** |

"Robust" models are trained with *station dropout*: about 15% of stations have their whole input window
hidden in each training sample. The LSTM can only fall back on time-of-day / season patterns, while the
GAT learns to rebuild the missing station from its neighbours (**+1h error 24.5 vs 41.1 Âµg/mÂ³**), at no cost
to normal accuracy.

---

## Final models for live use

The models above were evaluated on two unseen years. For deployment, the same architectures were
**retrained once and finally trained on all available data, Jan 2023 â€“ Sep 2026** (three winters).
Jul â€“ Sep 2026 was first held out to choose the stopping point, then added for a last short training
pass (3 epochs, decaying learning rate). They are never
retrained during use: each forecast just reads the latest live readings. Because the final models have
now seen every month we have, their honest test is the live scoring in the app; the hold-out results
above remain the evidence that the method generalises.

**Forecasting right after start-up.** The live feed only provides the current hour, so when the app is
opened without a collector running the model sees little history. A one-time fine-tuning with *history
dropout* (in half the training samples only the last 1â€“23 hours are visible) fixes this.
MAE on Jul â€“ Sep 2026, Âµg/mÂ³, measured before the last all-data pass (while those months were still
held out; the monsoon season, and the period used to pick the best fine-tuning round, so slightly
optimistic):

| Model | 24 h history | 6 h | 3 h | **1 h** |
|---|---|---|---|---|
| Persistence | 16.4 | 16.4 | 16.4 | 16.4 |
| GAT-Transformer, before fine-tuning | 13.6 | 15.9 | 16.1 | 16.3 |
| **GAT-Transformer, fine-tuned** | **13.6** | **14.0** | **14.2** | **14.2** |
| Robust GAT-Transformer, fine-tuned | 13.6 | 14.1 | 14.2 | 14.3 |
| Transformer, fine-tuned | 13.7 | 14.2 | 14.3 | 14.4 |
| LSTM, fine-tuned | 13.8 | 14.8 | 15.7 | 14.7 |

Full-history accuracy is unchanged; with only the current hour the GAT-Transformer error drops from
16.3 (no better than persistence) to 14.2.

**Ensemble.** Live forecasts also include the average of GAT-Transformer, Robust GAT-Transformer and
Transformer. On the two-year held-out test it is ~1% better than the best single model
(MAE 26.54 vs 26.77 Âµg/mÂ³), and better or equal at 4 of 5 lead times.

**Weather was tested and left out.** Adding Open-Meteo weather (past weather plus the forecast issued a
day earlier, so no peeking at the future) to correct the GAT-Transformer's forecasts improved 6â€“24 h
forecasts by only ~2â€“3% on held-out months of 2025, and did not help at +1 h â€” recent PM2.5 already
reflects current weather. Open-Meteo weather itself agreed well with CPCB station sensors (temperature
r â‰ˆ 0.96, humidity r â‰ˆ 0.9, no time offset).

### Honest limitations
- **Winter is harder than summer** (MAE â‰ˆ 40 vs 16): smog episodes change fast, and only one full Octâ€“Dec season is in the training data.
- **+24h forecasts** are only modestly better than persistence; without weather inputs, day-ahead PM2.5 is fundamentally hard.
- **Live mode** uses WAQI's US-EPA (2016) AQI values converted back to Âµg/mÂ³ (capped near 500 Âµg/mÂ³). Two stations (ITO, Shadipur) are not in the WAQI feed and appear as offline.

---

## Data

| | |
|---|---|
| Sources | [XKDR Air Quality API](https://airquality.xkdr.org) (Jan 2023 â€“ Aug 2025); CPCB hourly station files via the public Kaggle mirror [*Delhi Air Quality: CPCB Station Data (2010â€“2026)*](https://www.kaggle.com/datasets/yug201/delhi-air-quality-cpcb-hourly-2010-2026) (Jan 2025 â€“ Sep 2026) |
| Stations | 36 Delhi stations with â‰¥ 70% real-data coverage |
| Period | 1 Jan 2023 â€“ 30 Sep 2026, hourly (32,856 hours) |
| Coverage | 93.5% of station-hours have a real reading |
| Cleaning | Non-physical values (â‰¤ 0 or > 1000) and stuck sensors (â‰¥ 12 identical consecutive hours) removed |

| Split | Period | Forecast origins |
|---|---|---|
| Train | Jan 2023 â€“ Jun 2024 | 12,905 |
| Validation | Jul â€“ Sep 2024 | 2,184 |
| Test | Oct 2024 â€“ Sep 2026 | 17,393 |

The split is **by time**: every station appears in every split, and the models are always tested on months
they never saw.

## Method

**Inputs per station per hour (8 features):** standardised PM2.5, an `observed` flag (1 = real reading,
0 = filled), and sine/cosine of hour-of-day, day-of-week and day-of-year. 24-hour lookback.

**No leakage:** per-station mean/std fitted on training data only; input gaps filled by forward-fill
(â‰¤ 24 h) then station mean, never using future values; loss and metrics ignore unobserved targets.

**Graph:** Haversine distances between stations; each station links to its 8 nearest neighbours with
weight `exp(-(d/Ïƒ)Â²)`, Ïƒ = std of pairwise distances (8.1 km). The GAT adds `log(weight)` to its attention
scores as a distance prior that it can learn to override.

**Models** (all neural models share station embeddings and residual forecasting, so the comparison is fair):

| Model | Idea | Parameters |
|---|---|---|
| Persistence | Last observed value | 0 |
| Seasonal naive | Same hour, previous day | 0 |
| LSTM | 2-layer LSTM per station | 72k |
| Transformer | 2-layer Transformer encoder per station | 74k |
| **GAT-Transformer** | Graph attention across stations at every hour â†’ temporal Transformer â†’ graph attention on the summary â†’ multi-horizon head | 91k |
| Robust variants | Same architectures + station dropout (p = 0.15) during training | â€” |

**Training:** AdamW (lr 1e-3, wd 1e-4), masked Huber loss, ReduceLROnPlateau, early stopping on
validation (patience 4, max 20 epochs), each epoch on a random 50% of training windows, 3 seeds per model.
Trained on CPU.

---

## Setup and usage

### 1. Requirements
- **Python 3.12** (tested with 3.12.12; 3.10 â€“ 3.12 should work)
- Windows, Linux or macOS. **No GPU needed** â€” everything runs on CPU.
- About 1 GB of disk space (mostly PyTorch).
- Libraries (all listed in [`requirements.txt`](requirements.txt)):

| Library | Version tested | Used for |
|---|---|---|
| numpy | 2.5.4 | arrays |
| pandas | 3.1.0 | data tables |
| torch (PyTorch) | 2.15.0 | deep-learning models |
| scikit-learn | 1.9.1 | metrics |
| scipy | 1.18.1 | significance test |
| matplotlib | 3.11.2 | notebook plots |
| streamlit | 1.67.0 | web app |
| plotly | 7.1.0 | interactive charts and map |
| pyarrow | 26.0.1 | fast data loading in the app |
| jupyter | 1.1.1 | notebook (optional) |

### 2. Install the dependencies
Run these in a terminal **inside the project folder**:

```powershell
# Windows (PowerShell)
python -m venv .venv                 # create a virtual environment
.venv\Scripts\activate               # activate it
pip install -r requirements.txt      # install all libraries
```

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If PowerShell blocks the activate script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or skip
activation and call the interpreter directly: `.venv\Scripts\python.exe ...`.

### 3. Dataset
**Nothing to download â€” the dataset is included in the repository:**

| File | What it is |
|---|---|
| `data/raw/delhi_pm25_2023_2025_raw.csv` | Hourly PM2.5, 36 stations, Jan 2023 â€“ Aug 2025 (XKDR Air Quality API) |
| `data/raw/delhi_pm25_cpcb_2025_2026.csv` | Hourly PM2.5, Jan 2025 â€“ Sep 2026 (CPCB files, Kaggle: [`yug201/delhi-air-quality-cpcb-hourly-2010-2026`](https://www.kaggle.com/datasets/yug201/delhi-air-quality-cpcb-hourly-2010-2026)) |
| `data/raw/delhi_stations_metadata.csv` | Station names and coordinates |
| `data/processed/dataset.npz` | Cleaned, model-ready data for the test run |
| `data/processed/dataset_final.npz` | Same, for the final (live) models |

**To download the data again yourself** (optional):
1. XKDR data needs a free API key: put `XKDR_API_KEY=your_key` in a `.env` file in the project folder
   (copy `.env.example` to `.env`).
2. Run:
   ```bash
   cd src
   python fetch_stations.py          # station list
   python fetch_measurements.py      # 2023 â€“ 2025 readings (XKDR, needs the key)
   python fetch_kaggle_cpcb.py       # 2025 â€“ 2026 readings (Kaggle CPCB files, no key or login needed)
   ```
   Files are saved to `data/raw/` with the names above. If you download the Kaggle dataset manually instead,
   just run `fetch_kaggle_cpcb.py` â€” it builds `delhi_pm25_cpcb_2025_2026.csv` for you.

**For live mode** (current readings), get a free token at
[aqicn.org/data-platform/token](https://aqicn.org/data-platform/token) and add `WAQI_TOKEN=your_token` to `.env`.
Without it the app still shows all test results; only the live tabs are empty.

### 4. Run the project (main file)
**Main file: [`app/streamlit_app.py`](app/streamlit_app.py)** â€” the trained models are included, so this works
right after installing:

```powershell
python -m streamlit run app/streamlit_app.py
```
Then open **http://localhost:8501** in a browser. Tabs:
- **ðŸ“¡ Today** â€“ live PM2.5 at every station and the forecast for the next 24 hours
- **ðŸŽ¯ Prediction vs reality** â€“ live forecasts compared with what the sensors then measured
- **ðŸ† How good is it?** â€“ test results on two years of unseen data
- **â„¹ï¸ About** â€“ plain-language explanation

**Collect live data** (needed for "Prediction vs reality"): double-click `scripts\start_collector.bat`
(Windows) or run `python src/live.py --loop`, and leave it running. It saves readings and forecasts every hour
to `data/live/`.

### 5. Train and test the models
All commands run from the `src` folder. Training is on CPU.

```bash
cd src

# (a) Prepare data
python preprocess.py              # clean, split by time, build features  -> data/processed/dataset.npz
python build_graph.py             # station graph (8 nearest neighbours)   -> data/processed/adjacency_matrix.npy

# (b) TRAIN all models (LSTM, Transformer, GAT-Transformer + robust versions), 3 seeds each, then TEST
python train_eval.py                       # ~5 h on CPU
python train_eval.py --train-frac 0.25     # faster, slightly lower accuracy

# (c) TEST only (uses the saved checkpoints, no training)
python train_eval.py --eval-only           # -> results/metrics_*.csv, dm_test.csv, test_predictions.npz
python sensor_failure_eval.py              # sensor-failure test -> results/sensor_failure.csv
```

**Final models for the live app** (trained on all data up to Sep 2026):
```powershell
# Windows PowerShell
$env:AQI_RUN = "final"
python preprocess.py
python train_eval.py --seeds 0 --train-frac 0.25 --no-eval --models LSTM Transformer GAT-Transformer "Robust GAT-Transformer"
python train_eval.py --seeds 0 --train-frac 0.25 --no-eval --epochs 8 --patience 3 --finetune --history-dropout 0.5 --models LSTM Transformer GAT-Transformer "Robust GAT-Transformer"
python train_eval.py --seeds 0 --train-frac 0.25 --no-eval --epochs 3 --train-on-all --history-dropout 0.5 --models LSTM Transformer GAT-Transformer "Robust GAT-Transformer"
Remove-Item Env:AQI_RUN
```
(Linux/macOS: put `AQI_RUN=final` in front of each command instead.)

Useful `train_eval.py` options: `--models` (which models), `--seeds` (e.g. `0 1 2`), `--epochs`, `--patience`,
`--train-frac` (share of training windows per epoch), `--threads` (CPU threads).

**Notebook:** `Spatiotemporal_AQI_Forecasting.ipynb` walks through the data, graph, models and results
(`jupyter notebook` to open it).

### 6. Folder structure

```text
AQI/
â”œâ”€â”€ app/
â”‚   â””â”€â”€ streamlit_app.py            # â˜… MAIN FILE â€“ the web app
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ config.py                   # settings: paths, dates, forecast hours, AQI levels
â”‚   â”œâ”€â”€ fetch_stations.py           # download station list (XKDR)
â”‚   â”œâ”€â”€ fetch_measurements.py       # download hourly PM2.5 2023-2025 (XKDR)
â”‚   â”œâ”€â”€ fetch_kaggle_cpcb.py        # download hourly PM2.5 2025-2026 (Kaggle CPCB)
â”‚   â”œâ”€â”€ preprocess.py               # cleaning, time split, features
â”‚   â”œâ”€â”€ build_graph.py              # station graph
â”‚   â”œâ”€â”€ data.py                     # loads the dataset, makes 24-h input windows
â”‚   â”œâ”€â”€ models.py                   # baselines, LSTM, Transformer, GAT-Transformer
â”‚   â”œâ”€â”€ train_eval.py               # TRAINING (and calls the test)
â”‚   â”œâ”€â”€ evaluate.py                 # TESTING: metrics, significance test
â”‚   â”œâ”€â”€ sensor_failure_eval.py      # sensor-failure test
â”‚   â””â”€â”€ live.py                     # live readings, live forecasts, collector
â”œâ”€â”€ scripts/
â”‚   â””â”€â”€ start_collector.bat         # double-click to collect live data (Windows)
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ raw/                        # downloaded dataset (CSV)
â”‚   â”œâ”€â”€ processed/                  # model-ready dataset, graph, scaler, station list
â”‚   â””â”€â”€ live/                       # live readings and forecasts (created when collecting)
â”œâ”€â”€ checkpoints/                    # trained models from the test run (<model>_seed<k>.pt)
â”‚   â””â”€â”€ final/                      # final models used by the live app
â”œâ”€â”€ results/                        # test metrics (CSV), test predictions, training logs
â”œâ”€â”€ Spatiotemporal_AQI_Forecasting.ipynb   # walkthrough notebook
â”œâ”€â”€ MODEL_SUMMARY.txt               # one-page summary of data, factors and results
â”œâ”€â”€ requirements.txt                # Python libraries
â”œâ”€â”€ .env.example                    # template for API keys (copy to .env)
â””â”€â”€ README.md
```

---

## Demo script (5 minutes)

1. **ðŸ“¡ Today** â€“ Delhi's air right now on a map, health advice, and the next 24 hours for any station.
2. **Sensor failure** â€“ in the sidebar switch on *Pretend this station's sensor is broken*: the GAT model still
   forecasts the station from its neighbours.
3. **ðŸŽ¯ Prediction vs reality** â€“ what the model predicted over the last few days vs what really happened.
4. **ðŸ† How good is it?** â€“ the scorecard on two unseen years (Oct 2024 â€“ Sep 2026) and the comparison of models.

## Acknowledgements
Central Pollution Control Board (CPCB), Delhi Pollution Control Committee (DPCC) and India
Meteorological Department (IMD) for the monitoring network; XKDR Forum for the Air Quality API;
Yug Kaushik for the CPCB Kaggle mirror; the World Air Quality Index project (aqicn.org) for the live feed.
