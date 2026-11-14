# Delhi PM2.5 Forecaster
### Spatiotemporal air-quality forecasting with Graph Attention + Transformer networks

Forecasts hourly PM2.5 at **36 monitoring stations across Delhi** for the next **1, 3, 6, 12 and 24 hours**.
Trained on real measurements from Jan 2023 – Jun 2024 and tested on **two full unseen years
(Oct 2024 – Sep 2026)**. Comes with an interactive **Streamlit app** that shows **live PM2.5 for Delhi
right now**, forecasts the next 24 hours, and scores those live forecasts as the real readings arrive.

---

## Highlights

- **Real data only.** Hourly PM2.5 from 36 CPCB / DPCC / IMD stations: XKDR Air Quality API (2023 – Aug 2025) and the CPCB hourly files (2025 – Sep 2026). The two sources agree exactly on all 117,055 overlapping readings. No synthetic values.
- **Graph neural network.** Stations form a graph (8 nearest neighbours, distance-weighted). A Graph Attention Network learns, hour by hour, which neighbouring stations matter.
- **Rigorous evaluation.** Chronological split, scalers fitted on training data only, causal (past-only) gap filling, metrics on real observations only, 3 random seeds per model, Diebold–Mariano significance tests, and a test set that runs **more than two years past the training data**.
- **Robust to sensor failure.** The *Robust GAT-Transformer* forecasts a station whose sensor is down with **only +17% error** (vs +48% for an equally trained LSTM), by learning from its neighbours.
- **Live mode, no maintenance.** Final models trained on all data up to Sep 2026 and fine-tuned once to forecast from just the current live readings. Open the app any time: it fetches live PM2.5, forecasts the next 24 hours, saves the forecast and scores it when the real readings arrive. No daily retraining.

---

## Results

Test period: **Oct 2024 – Sep 2026** (17,393 hourly forecast origins × 36 stations), never seen in
training. Real observations only. Neural models: mean ± std over 3 seeds. MAE/RMSE in µg/m³.

| Model | MAE ↓ | RMSE ↓ | R² ↑ | AQI-category accuracy ↑ |
|---|---|---|---|---|
| Persistence (last value) | 36.06 | 63.33 | 0.566 | 51.2% |
| Seasonal naive (same hour yesterday) | 38.77 | 66.76 | 0.518 | 47.8% |
| LSTM | 27.65 ± 0.01 | 48.77 ± 0.14 | 0.743 | 56.8% |
| Transformer | 27.35 ± 0.22 | 47.82 ± 0.23 | **0.753** | 57.2% |
| **GAT-Transformer** | **26.94 ± 0.10** | 47.95 ± 0.33 | 0.751 | **57.8%** |
| Robust LSTM | 27.61 ± 0.16 | 48.60 ± 0.30 | 0.744 | 56.9% |
| Robust GAT-Transformer | 26.96 ± 0.29 | **47.81 ± 0.37** | **0.753** | 57.6% |

**MAE by lead time**

| Model | +1h | +3h | +6h | +12h | +24h |
|---|---|---|---|---|---|
| Persistence | 16.02 | 30.91 | 43.32 | 51.33 | 38.74 |
| LSTM | 14.74 | 24.80 | 30.27 | 33.69 | 34.79 |
| Transformer | 14.72 | 24.43 | 29.88 | 33.06 | 34.66 |
| **GAT-Transformer** | **14.34** | **23.65** | 29.25 | **32.85** | 34.63 |
| Robust GAT-Transformer | 14.43 | 23.68 | **29.23** | 32.98 | **34.54** |

**MAE by period / season**

| Model | Test year 1 (Oct 24–Sep 25) | Test year 2 (Oct 25–Sep 26) | **2026 only** | Winter (Oct–Feb) | Summer/Monsoon (Apr–Sep) | Severe (> 250) |
|---|---|---|---|---|---|---|
| Persistence | 35.47 | 36.67 | 29.10 | 54.40 | 20.90 | 106.97 |
| LSTM | 27.56 | 27.74 | 22.46 | 41.81 | 16.56 | 93.30 |
| Transformer | 27.25 | 27.45 | 22.27 | 41.06 | 16.57 | **90.30** |
| **GAT-Transformer** | 26.86 | **27.02** | **21.96** | 40.53 | **16.28** | 92.13 |
| Robust GAT-Transformer | **26.76** | 27.18 | 22.04 | **40.49** | 16.33 | 91.40 |

- **About 25% lower error than persistence**, and **no decline in accuracy in the second test year**: in 2026 the model is still as good as right after training.
- **Significance (Diebold–Mariano):** the GAT-Transformer beats the LSTM at +1h, +3h, +6h and +12h, and the Transformer at +1h, +3h and +6h (all p < 0.001). At +24h the neural models are statistically tied.

### Sensor-failure test

Each station's own last 24 h of readings is hidden in turn; the model must forecast that station anyway.

| Model | MAE, sensor working | MAE, sensor failed | Error increase |
|---|---|---|---|
| Station long-term average | — | 69.8 | — |
| LSTM | 27.6 | 52.6 | +90% |
| Transformer | 27.3 | 54.2 | +98% |
| GAT-Transformer | 26.9 | 51.3 | +91% |
| Robust LSTM | 27.6 | 40.9 | +48% |
| **Robust GAT-Transformer** | **26.8** | **31.4** | **+17%** |

"Robust" models are trained with *station dropout*: about 15% of stations have their whole input window
hidden in each training sample. The LSTM can only fall back on time-of-day / season patterns, while the
GAT learns to rebuild the missing station from its neighbours (**+1h error 24.5 vs 41.1 µg/m³**), at no cost
to normal accuracy.

---

## Final models for live use

The models above were evaluated on two unseen years. For deployment, the same architectures were
**retrained once and finally trained on all available data, Jan 2023 – Sep 2026** (three winters).
Jul – Sep 2026 was first held out to choose the stopping point, then added for a last short training
pass (3 epochs, decaying learning rate). They are never
retrained during use: each forecast just reads the latest live readings. Because the final models have
now seen every month we have, their honest test is the live scoring in the app; the hold-out results
above remain the evidence that the method generalises.

**Forecasting right after start-up.** The live feed only provides the current hour, so when the app is
opened without a collector running the model sees little history. A one-time fine-tuning with *history
dropout* (in half the training samples only the last 1–23 hours are visible) fixes this.
MAE on Jul – Sep 2026, µg/m³, measured before the last all-data pass (while those months were still
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
(MAE 26.54 vs 26.77 µg/m³), and better or equal at 4 of 5 lead times.

**Weather was tested and left out.** Adding Open-Meteo weather (past weather plus the forecast issued a
day earlier, so no peeking at the future) to correct the GAT-Transformer's forecasts improved 6–24 h
forecasts by only ~2–3% on held-out months of 2025, and did not help at +1 h — recent PM2.5 already
reflects current weather. Open-Meteo weather itself agreed well with CPCB station sensors (temperature
r ≈ 0.96, humidity r ≈ 0.9, no time offset).

### Honest limitations
- **Winter is harder than summer** (MAE ≈ 40 vs 16): smog episodes change fast, and only one full Oct–Dec season is in the training data.
- **+24h forecasts** are only modestly better than persistence; without weather inputs, day-ahead PM2.5 is fundamentally hard.
- **Live mode** uses WAQI's US-EPA (2016) AQI values converted back to µg/m³ (capped near 500 µg/m³). Two stations (ITO, Shadipur) are not in the WAQI feed and appear as offline.

---

## Data

| | |
|---|---|
| Sources | [XKDR Air Quality API](https://airquality.xkdr.org) (Jan 2023 – Aug 2025); CPCB hourly station files via the public Kaggle mirror [*Delhi Air Quality: CPCB Station Data (2010–2026)*](https://www.kaggle.com/datasets/yug201/delhi-air-quality-cpcb-hourly-2010-2026) (Jan 2025 – Sep 2026) |
| Stations | 36 Delhi stations with ≥ 70% real-data coverage |
| Period | 1 Jan 2023 – 30 Sep 2026, hourly (32,856 hours) |
| Coverage | 93.5% of station-hours have a real reading |
| Cleaning | Non-physical values (≤ 0 or > 1000) and stuck sensors (≥ 12 identical consecutive hours) removed |

| Split | Period | Forecast origins |
|---|---|---|
| Train | Jan 2023 – Jun 2024 | 12,905 |
| Validation | Jul – Sep 2024 | 2,184 |
| Test | Oct 2024 – Sep 2026 | 17,393 |

The split is **by time**: every station appears in every split, and the models are always tested on months
they never saw.

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

```powershell
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

python -m streamlit run app/streamlit_app.py
```

Open http://localhost:8501. The processed dataset and trained checkpoints are included, so the app runs
immediately.

### Live mode
1. Get a free personal token at [aqicn.org/data-platform/token](https://aqicn.org/data-platform/token).
2. Add `WAQI_TOKEN=your_token` to a `.env` file in the project folder (see `.env.example`).
3. Open the app. It fetches the current readings itself, forecasts the next 24 hours and saves the forecast.
4. Optional: keep the collector running so history builds up and every hour's forecast gets scored:
   ```powershell
   python src/live.py --loop
   ```
The **📡 Today** tab shows current readings, the next-24-h forecast for every station, and how
accurate the live forecasts have been so far. Live forecasts come from the final models in
`checkpoints/final/`; the evaluated models in `checkpoints/` back the "How good is it?" tab.

### Re-running the pipeline

```bash
cd src
python fetch_measurements.py      # XKDR data 2023-2025 (needs XKDR_API_KEY in .env)
python fetch_kaggle_cpcb.py       # CPCB files 2025-2026 (no key needed)
python preprocess.py              # clean, split, build features
python build_graph.py             # station graph
python train_eval.py              # train all models x 3 seeds, then evaluate   (~5 h on CPU)
python train_eval.py --eval-only  # re-evaluate existing checkpoints
python sensor_failure_eval.py     # sensor-failure test

# Final models for live use: train to Jun 2026 (Jul-Sep 2026 held out for early stopping), fine-tune
# once for short history, then a last pass on ALL data up to Sep 2026
AQI_RUN=final python preprocess.py
AQI_RUN=final python train_eval.py --seeds 0 --train-frac 0.25 --no-eval \
    --models LSTM Transformer GAT-Transformer "Robust GAT-Transformer"
AQI_RUN=final python train_eval.py --seeds 0 --train-frac 0.25 --no-eval --epochs 8 --patience 3 \
    --finetune --history-dropout 0.5 --models LSTM Transformer GAT-Transformer "Robust GAT-Transformer"
AQI_RUN=final python train_eval.py --seeds 0 --train-frac 0.25 --no-eval --epochs 3 \
    --train-on-all --history-dropout 0.5 --models LSTM Transformer GAT-Transformer "Robust GAT-Transformer"
```
(On Windows PowerShell set the variable first: `$env:AQI_RUN = "final"`.)

---

## Demo script (5 minutes)

No preparation needed: the app works from the current live readings. (If the collector has been running for a while, the live accuracy table will already have scores to show.)

1. **Today** – Delhi's air quality right now on a map, in plain language: colour-coded categories, health advice, and what to expect over the next 24 hours for any station you click.
2. **Live accuracy** – scroll down: *"How accurate have the live forecasts been?"* scores the forecasts made over the past day against the real readings that arrived since.
3. **Sensor failure, live** – switch on *Simulate sensor failure*. The *live sensor-failure check* shows each model's estimate of the station's current value made **without its own data**, next to the real reading: the LSTM drifts, the **Robust GAT-Transformer** stays close by using the neighbours. The bar chart shows which neighbours it relied on.
4. **How good is it?** – the proof: results on two unseen years (Oct 2024 – Sep 2026), significance tests, the sensor-failure test across all stations, and the per-station error map.

---

## Repository structure

```text
AQI/
├── app/
│   └── streamlit_app.py            # interactive demo (live + historical)
├── src/
│   ├── config.py                   # paths, dates, horizons, AQI bands
│   ├── fetch_stations.py           # station directory (XKDR API)
│   ├── fetch_measurements.py       # real hourly PM2.5, 2023-2025 (XKDR)
│   ├── fetch_kaggle_cpcb.py        # real hourly PM2.5, 2025-2026 (CPCB files)
│   ├── preprocess.py               # cleaning, split, features, forecast windows
│   ├── build_graph.py              # station graph
│   ├── data.py                     # dataset loading / window slicing
│   ├── models.py                   # baselines, LSTM, Transformer, GAT-Transformer
│   ├── train_eval.py               # multi-seed training
│   ├── evaluate.py                 # test metrics, significance tests
│   ├── sensor_failure_eval.py      # sensor-failure test
│   └── live.py                     # live readings (WAQI), live forecasts and scoring
├── data/
│   ├── raw/                        # station metadata + raw measurements
│   └── processed/                  # dataset.npz, graph, scaler, station list, WAQI station map
├── checkpoints/                    # evaluated models (<model>_seed<k>.pt)
│   └── final/                      # live models, trained on all data to Sep 2026
├── results/                        # metrics CSVs, test predictions, training logs
│   └── final/                      # final models' training logs
├── Spatiotemporal_AQI_Forecasting.ipynb   # walkthrough notebook
└── requirements.txt
```

## Acknowledgements
Central Pollution Control Board (CPCB), Delhi Pollution Control Committee (DPCC) and India
Meteorological Department (IMD) for the monitoring network; XKDR Forum for the Air Quality API;
Yug Kaushik for the CPCB Kaggle mirror; the World Air Quality Index project (aqicn.org) for the live feed.
