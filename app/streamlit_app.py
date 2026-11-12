"""
Delhi PM2.5 Forecaster — live demo.

Run from the project root:
    python -m streamlit run app/streamlit_app.py
Live data needs WAQI_TOKEN in .env; keep `python src/live.py --loop` running to build up history.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
# Live forecasts use the final models (trained on all data up to Sep 2026) once they exist; the
# "How good is the model?" tab always shows the evaluated (hold-out) models' test results.
if "AQI_RUN" not in os.environ and os.path.exists(os.path.join(ROOT, "results", "final", "best_seeds.json")):
    os.environ["AQI_RUN"] = "final"

import config as C  # noqa: E402
from data import AQIData  # noqa: E402
from models import MODEL_CLASSES, Persistence, build_model, slug  # noqa: E402
import live as L  # noqa: E402

st.set_page_config(page_title="Delhi PM2.5 Forecaster", page_icon="🌫️", layout="wide")

MODEL_COLORS = {
    "Persistence": "#9CA3AF", "Seasonal naive": "#C4B5FD", "LSTM": "#3B82F6",
    "Transformer": "#F59E0B", "GAT-Transformer": "#10B981",
    "Robust LSTM": "#1E3A8A", "Robust GAT-Transformer": "#047857", "Ensemble": "#7C3AED",
}
BAND_COLORS = {name: color for _, _, name, color in C.AQI_BANDS}


# ----------------------------------------------------------------------------- loading
def _version(*paths):
    """Changes whenever any of the files/folders change, so cached data is reloaded automatically."""
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += [os.path.getmtime(os.path.join(p, f)) for f in sorted(os.listdir(p))]
        elif os.path.exists(p):
            out.append(os.path.getmtime(p))
    return tuple(out)


@st.cache_resource(show_spinner="Loading models…")
def load_all(_version):
    data = AQIData()
    best_path = os.path.join(C.RESULTS_DIR, "best_seeds.json")
    best = json.load(open(best_path)) if os.path.exists(best_path) else {}
    models = {}
    for name in MODEL_CLASSES:
        path = os.path.join(C.CHECKPOINTS_DIR, f"{slug(name)}_seed{best.get(name, '0')}.pt")
        if os.path.exists(path):
            m = build_model(name, data.n_nodes, len(C.HORIZONS), window=C.INPUT_WINDOW)
            m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            models[name] = m.eval()
    return data, models


@st.cache_data
def load_results(_version):
    def rd(name):
        p = os.path.join(C.HOLDOUT_RESULTS_DIR, name)
        return pd.read_csv(p) if os.path.exists(p) else None
    out = {k: rd(f"{k}.csv") for k in ["metrics_overall", "metrics_horizon", "metrics_season",
                                         "metrics_station", "dm_test", "sensor_failure"]}
    logs = {}
    log_dir = os.path.join(C.HOLDOUT_RESULTS_DIR, "train_logs")
    if os.path.isdir(log_dir):
        for fn in sorted(os.listdir(log_dir)):
            with open(os.path.join(log_dir, fn)) as f:
                logs[fn[:-5]] = json.load(f)
    out["logs"] = logs
    return out


data, models = load_all(_version(C.DATASET_NPZ, C.CHECKPOINTS_DIR, C.STATIONS_CSV))
res = load_results(_version(C.HOLDOUT_RESULTS_DIR, os.path.join(C.HOLDOUT_RESULTS_DIR, "train_logs")))
stations = data.stations
names = stations["name"].tolist()


def aqi_color(v):
    return BAND_COLORS[C.aqi_category(v)] if np.isfinite(v) else "#D1D5DB"


def add_aqi_bands(fig, ymax):
    for lo, hi, name, color in C.AQI_BANDS:
        if lo > ymax:
            break
        fig.add_hrect(y0=lo, y1=min(hi, ymax), fillcolor=color, opacity=0.08, line_width=0,
                      annotation_text=name, annotation_position="right",
                      annotation_font_size=10, annotation_font_color="#6B7280")


@torch.no_grad()
def live_forecast(readings, end, hide=None):
    """Run every model on the live 24 h window ending at `end`.
    hide: station index whose sensor is simulated as failed. Returns ({model: [H, N]}, attention)."""
    X, _ = L.build_window(readings, data, end)
    if hide is not None:
        X = data.hide_stations(X, [hide])
    out = {"Persistence": data.denorm(Persistence(C.HORIZONS)(X).numpy()[0])}
    attn = {}
    for name, m in models.items():
        if "GAT" in name:
            y, a = m(X, data.adj, return_attn=True)
            attn[name] = a[0].numpy()
        else:
            y = m(X, data.adj)
        out[name] = data.denorm(y.numpy()[0])
    return L.add_ensemble(out), attn


# ----------------------------------------------------------------------------- sidebar
# A click on the map selects a station (handled before the selectbox is drawn).
ev = st.session_state.get("today_map")
if ev and ev.get("selection", {}).get("points"):
    picked = ev["selection"]["points"][0]["customdata"][0]
    if picked != st.session_state.get("_last_map_pick"):
        st.session_state["_last_map_pick"] = picked
        st.session_state["station"] = picked
st.session_state.setdefault("station", "Anand Vihar")

with st.sidebar:
    st.title("🌫️ Delhi PM2.5 Forecaster")
    st.caption(f"{len(stations)} Delhi monitoring stations · live data via the World Air Quality Index project · "
               + ("live forecasts by the final models, trained on all data, Jan 2023 – Sep 2026" if C.RUN == "final" else
                  "live forecasts by the evaluated models, trained on Jan 2023 – Jun 2024"))
    station = st.selectbox("Station", names, key="station")
    sidx = names.index(station)
    failed = st.toggle("🔌 Simulate sensor failure", key="failed",
                       help="Hide this station's own live readings from every model, as if its sensor "
                            "were broken. Models that only look at the station itself are left guessing; "
                            "the GAT models can still use neighbouring stations.")
    default_models = (["Persistence", "LSTM", "Robust GAT-Transformer"] if failed
                      else ["Persistence", "LSTM", "GAT-Transformer", "Ensemble"])
    available = ["Persistence"] + [m for m in MODEL_COLORS if m in models] + (
        ["Ensemble"] if all(m in models for m in L.ENSEMBLE_MEMBERS) else [])
    # Keyed on `failed` so the selection resets to sensible defaults when the toggle flips.
    shown = st.multiselect("Models to show", available, key=f"shown_{failed}",
                           default=[m for m in default_models if m in available])
    st.divider()
    st.caption("Tip: click a station on the map to select it.")
    if not models:
        st.error("No trained checkpoints found — run `python src/train_eval.py` first.")

tab_today, tab_cmp, tab_about = st.tabs(["📡 Today (live)", "🏆 How good is the model?", "ℹ️ About"])

# ----------------------------------------------------------------------------- today (live)
with tab_today:
    token = L.load_token()
    if not token:
        st.info("Live mode needs a free WAQI token: get one at aqicn.org/data-platform/token and add "
                "`WAQI_TOKEN=...` to the `.env` file in the project folder.")
        st.stop()

    head, btn = st.columns([5, 1])
    stale = (not os.path.exists(L.READINGS_CSV) or
             pd.Timestamp.now().timestamp() - os.path.getmtime(L.READINGS_CSV) > 20 * 60)
    if btn.button("🔄 Fetch now", width="stretch") or stale:
        with st.spinner("Fetching live readings from Delhi's stations…"):
            try:
                L.collect(data, models, token)
            except Exception as e:
                st.warning(f"Could not fetch live data right now ({e}). Showing the last saved readings.")
    readings, fcs = L.load_readings(), L.load_forecasts()
    if readings.empty:
        st.warning("No live readings yet — press “Fetch now”.")
        st.stop()

    latest = readings["time"].max()
    recent = readings[readings["time"] >= latest - pd.Timedelta(hours=2)]
    now_by_station = recent.sort_values("time").groupby("station_id")["pm25"].last()
    now_vals = stations["station_id"].map(now_by_station).values.astype(float)
    hours_have = readings.loc[readings["time"] > latest - pd.Timedelta(hours=C.INPUT_WINDOW), "time"].nunique()
    head.markdown(f"#### Delhi right now — {latest:%d %b %Y, %H:00} IST  \n"
                  f"Live CPCB/DPCC readings · **{int(np.isfinite(now_vals).sum())} of {len(stations)} "
                  f"stations reporting** · history collected: **{hours_have} of 24 hours**")
    if hours_have < C.INPUT_WINDOW:
        st.info(f"Forecasting from {hours_have} h of collected history (the models were built for 24 h). "
                f"They still work from just the current readings — in tests, error is about 12% higher with "
                f"one hour of history than with a full day, and still ~15% better than persistence. "
                f"Keep `python src/live.py --loop` running and forecasts sharpen as history builds up.")

    fc, attn = live_forecast(readings, latest, hide=sidx if failed else None)
    if failed:
        st.error(f"🔌 **Sensor failure simulated at {station}.** No model can see this station's own "
                 f"readings; they must work it out from neighbouring stations. (With no data of its own, "
                 f"*Persistence* falls back to the station's long-term average.)")

    col_m, col_c = st.columns([3, 2], gap="large")
    with col_m:
        tdf = stations.assign(
            pm25=np.round(now_vals, 0),
            category=[C.aqi_category(v) if np.isfinite(v) else "Offline" for v in now_vals],
            selected=[n == station for n in names])
        tdf["size"] = np.where(tdf["selected"], 22, 13)
        fig = px.scatter_map(
            tdf, lat="lat", lon="lon", color="category", size="size", size_max=22,
            color_discrete_map={**BAND_COLORS, "Offline": "#D1D5DB"},
            category_orders={"category": [b[2] for b in C.AQI_BANDS] + ["Offline"]},
            custom_data=["name", "pm25"], hover_name="name",
            hover_data={"pm25": True, "category": True, "lat": False, "lon": False, "size": False},
            zoom=9.4, center={"lat": 28.65, "lon": 77.17}, height=500, map_style="carto-positron")
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend_title_text="AQI category",
                          legend=dict(orientation="h", y=-0.02))
        st.plotly_chart(fig, width="stretch", key="today_map", on_select="rerun", selection_mode="points")

    with col_c:
        cur = now_vals[sidx]
        st.markdown(
            f"<div style='border-radius:12px;padding:16px 18px;background:{aqi_color(cur)}22;"
            f"border-left:6px solid {aqi_color(cur)}'>"
            f"<div style='font-size:13px;color:#6B7280'>{station} — PM2.5 now"
            f"{' (hidden from models)' if failed else ''}</div>"
            f"<div style='font-size:40px;font-weight:700'>{'—' if not np.isfinite(cur) else f'{cur:.0f}'}"
            f" <span style='font-size:16px;font-weight:400'>µg/m³</span></div>"
            f"<div style='font-size:16px;font-weight:600'>"
            f"{C.aqi_category(cur) if np.isfinite(cur) else 'Station offline'}</div></div>",
            unsafe_allow_html=True)
        rows = []
        for hi, h in enumerate(C.HORIZONS):
            row = {"Lead": f"+{h}h", "Time": f"{latest + pd.Timedelta(hours=h):%d %b %H:00}"}
            row.update({n: round(float(fc[n][hi, sidx])) for n in shown if n in fc})
            rows.append(row)
        st.markdown(f"**Forecast from {latest:%H:00} (µg/m³)**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if not np.isfinite(cur) and not failed:
            st.caption("This station is offline right now: its forecast comes from neighbouring stations "
                       "(the Robust GAT-Transformer is built for exactly this).")

    # Live readings + the next 24 hours.
    st_read = readings[readings["station_id"] == stations["station_id"].iloc[sidx]].sort_values("time")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=st_read["time"], y=st_read["pm25"], mode="lines+markers",
                             name="Observed (hidden from models)" if failed else "Observed (live)",
                             line=dict(color="#9CA3AF" if failed else "#111827", width=2.5)))
    for name in [n for n in shown if n in fc]:
        fig.add_trace(go.Scatter(x=[latest + pd.Timedelta(hours=h) for h in C.HORIZONS],
                                 y=fc[name][:, sidx], name=name, mode="lines+markers",
                                 line=dict(color=MODEL_COLORS.get(name), width=2.5 if "GAT" in name else 1.8)))
    yvals = list(st_read["pm25"].dropna()) + [float(fc[n][:, sidx].max()) for n in shown if n in fc]
    ymax = max(yvals) * 1.15 if yvals else 300
    add_aqi_bands(fig, ymax)
    fig.add_vline(x=latest, line_dash="dash", line_color="#6B7280")
    fig.update_layout(height=400, yaxis_title="PM2.5 (µg/m³)", yaxis_range=[0, ymax], hovermode="x unified",
                      margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.12),
                      title=dict(text=f"{station}: live readings and the next 24 hours", font_size=15))
    st.plotly_chart(fig, width="stretch")

    # Live sensor-failure check: estimate the current hour without the station's own data.
    if failed:
        prev_end = latest - pd.Timedelta(hours=1)
        if not np.isfinite(cur) or readings["time"].min() > prev_end:
            st.caption("Live check unavailable: needs this station's current reading and at least one "
                       "earlier hour of data.")
        else:
            fc_prev, _ = live_forecast(readings[readings["time"] <= prev_end], prev_end, hide=sidx)
            chk = pd.DataFrame([{"Model": n, "Estimate for now (µg/m³)": round(float(fc_prev[n][0, sidx])),
                                 "Error vs real reading": round(abs(float(fc_prev[n][0, sidx]) - cur))}
                                for n in fc_prev if n in shown or "GAT" in n]).sort_values("Error vs real reading")
            st.markdown(f"**Live sensor-failure check.** Pretend {station}'s sensor broke an hour ago: "
                        f"each model estimates the value for {latest:%H:00} without any of the station's own "
                        f"data. The real reading is **{cur:.0f} µg/m³**.")
            st.dataframe(chk, hide_index=True, width="stretch")

    attn_model = "Robust GAT-Transformer" if failed and "Robust GAT-Transformer" in attn else "GAT-Transformer"
    if attn_model in attn:
        st.markdown(f"**Which neighbouring stations is the {attn_model} listening to right now?** "
                    f"Graph-attention weights for {station}.")
        w = attn[attn_model][sidx]
        nb = np.where(w > 0.005)[0]
        adf = pd.DataFrame({"Station": [names[j] + (" (itself)" if j == sidx else "") for j in nb],
                            "Attention": w[nb], "Distance (km)": data.dist[sidx, nb].round(1),
                            "PM2.5 now": np.round(now_vals[nb], 0)}).sort_values("Attention")
        fig = px.bar(adf, x="Attention", y="Station", orientation="h",
                     hover_data=["Distance (km)", "PM2.5 now"], color_discrete_sequence=["#10B981"])
        fig.update_layout(height=max(220, 28 * len(adf)), margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_tickformat=".0%")
        st.plotly_chart(fig, width="stretch")

    st.markdown("**How accurate have the live forecasts been?**  Every hour each model's forecast is saved; "
                "once the target hour arrives it is scored against the real reading. A +1h forecast is "
                "scored an hour later, +6h after six hours, and so on.")
    sc = L.live_scores(readings, fcs)
    if sc.empty:
        st.caption("Nothing to score yet — the first forecast reaches its target hour (+1h) within an hour.")
    else:
        piv = sc.pivot(index="model", columns="horizon", values="MAE").round(1)
        piv.columns = [f"+{h}h MAE" for h in piv.columns]
        piv["forecasts scored"] = sc.groupby("model")["n"].sum()
        st.dataframe(piv.sort_values(piv.columns[0]), width="stretch")

# ----------------------------------------------------------------------------- how good is the model
with tab_cmp:
    ov = res.get("metrics_overall")
    if ov is None:
        st.info("Run the evaluation to see model comparison.")
    else:
        st.markdown("#### Accuracy on two years the models never saw — all 36 stations")
        st.caption("Test period: Oct 2024 – Sep 2026 (incl. two winters), real observations only. "
                   "Neural models: mean ± std over 3 random seeds.")
        best = ov.loc[ov["MAE_mean"].idxmin()]
        k = st.columns(4)
        k[0].metric("Best model (MAE)", best["Model"])
        k[1].metric("MAE", f"{best['MAE_mean']:.1f} µg/m³")
        pers = ov.loc[ov["Model"] == "Persistence", "MAE_mean"].iloc[0]
        k[2].metric("Error reduction vs persistence", f"{(1 - best['MAE_mean'] / pers):.1%}")
        k[3].metric("AQI category accuracy", f"{best['AQI_acc_mean']:.1%}")
        st.dataframe(ov[["Model", "MAE", "RMSE", "R²", "AQI category acc."]], hide_index=True,
                     width="stretch")

        hz = res["metrics_horizon"]
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            for name, g in hz.groupby("Model", sort=False):
                fig.add_trace(go.Scatter(x=[f"+{h}h" for h in g["Horizon"]], y=g["MAE_mean"],
                                         error_y=dict(type="data", array=g["MAE_std"]),
                                         name=name, mode="lines+markers",
                                         line=dict(color=MODEL_COLORS.get(name))))
            fig.update_layout(title="MAE by lead time", yaxis_title="MAE (µg/m³)", height=380,
                              margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, width="stretch")
        with c2:
            ss = res["metrics_season"]
            fig = px.bar(ss, x="Subset", y="MAE_mean", color="Model", barmode="group",
                         error_y="MAE_std", color_discrete_map=MODEL_COLORS)
            fig.update_layout(title="MAE by period / season", yaxis_title="MAE (µg/m³)",
                              xaxis_title=None, height=380, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, width="stretch")

        sf = res.get("sensor_failure")
        if sf is not None:
            st.markdown("**Sensor-failure test** — each station's own readings hidden in turn; "
                        "the model must forecast it from neighbouring stations only.")
            c1, c2 = st.columns([3, 2])
            with c1:
                fig = go.Figure()
                for name, g in sf.groupby("Model", sort=False):
                    fig.add_trace(go.Bar(x=[f"+{h}h" for h in g["Horizon"]], y=g["MAE_failed"], name=name,
                                         marker_color=MODEL_COLORS.get(name, "#D1D5DB")))
                fig.update_layout(barmode="group", yaxis_title="MAE with sensor failed (µg/m³)",
                                  height=360, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, width="stretch")
            with c2:
                tbl = sf.groupby("Model", sort=False)[["MAE_normal", "MAE_failed"]].mean()
                tbl["Error increase"] = (tbl["MAE_failed"] / tbl["MAE_normal"] - 1).map(
                    lambda v: "—" if not np.isfinite(v) else f"{v:+.0%}")
                st.dataframe(tbl.rename(columns={"MAE_normal": "MAE, sensor working",
                                                 "MAE_failed": "MAE, sensor failed"}).round(1),
                             width="stretch")

        dm = res.get("dm_test")
        if dm is not None and len(dm):
            st.markdown("**Is the GAT-Transformer significantly better?** Diebold–Mariano test "
                        "(negative statistic = GAT-Transformer has lower error; p < 0.05 = significant).")
            piv = dm.assign(cell=[f"{s:+.2f} (p={p:.3f})" for s, p in zip(dm["DM stat"], dm["p-value"])]) \
                    .pivot(index="Comparison", columns="Horizon", values="cell")
            piv.columns = [f"+{c}h" for c in piv.columns]
            st.dataframe(piv, width="stretch")

        stm = res["metrics_station"]
        st.markdown("**Error per station**")
        mname = st.selectbox("Model", stm["Model"].unique(),
                             index=list(stm["Model"].unique()).index("GAT-Transformer")
                             if "GAT-Transformer" in stm["Model"].unique() else 0)
        sdf = stm[stm["Model"] == mname].merge(stations, on="station_id")
        c1, c2 = st.columns([3, 2])
        with c1:
            fig = px.scatter_map(sdf, lat="lat", lon="lon", color="MAE", size="MAE", hover_name="name",
                                 color_continuous_scale="Viridis_r", zoom=9.2,
                                 center={"lat": 28.65, "lon": 77.17}, height=420, map_style="carto-positron")
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, width="stretch")
        with c2:
            st.dataframe(sdf[["name", "MAE"]].sort_values("MAE").round(2), hide_index=True,
                         width="stretch", height=420)

        if res["logs"]:
            st.markdown("**Training curves** (masked Huber loss, standardised units)")
            rows = []
            for key, lg in res["logs"].items():
                for split in ("train", "val"):
                    for ep, v in enumerate(lg["history"][split], 1):
                        rows.append({"run": key, "split": split, "epoch": ep, "loss": v})
            fig = px.line(pd.DataFrame(rows), x="epoch", y="loss", color="run", line_dash="split", height=380)
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch")

# ----------------------------------------------------------------------------- about
with tab_about:
    st.markdown(f"""
#### How it works
**Live data.** Current hourly PM2.5 for Delhi's CPCB/DPCC stations from the World Air Quality Index
project (aqicn.org), converted from its US-EPA AQI values back to µg/m³. Readings are stored every hour;
each hour every model issues a forecast that is scored once the real reading arrives.

**Training data.** Real hourly PM2.5 from {len(stations)} Delhi monitoring stations (CPCB, DPCC, IMD):
XKDR Air Quality API (2023 – Aug 2025) and CPCB hourly files (2025 – Sep 2026). Trained on Jan 2023 – Jun 2024,
validated on Jul – Sep 2024, and tested on Oct 2024 – Sep 2026 — two full years never seen in training.

**No leakage.** Scalers are fitted on training data only; missing inputs are filled *causally*
(forward-fill only) with an `observed` flag as an extra input; models are trained and scored only on
real observed values.

**Graph.** Each station connects to its {C.KNN} nearest stations; edges weighted by a Gaussian kernel of
Haversine distance. The GAT uses log(weight) as an attention prior that it can learn to override.

**Models.** Persistence baseline; per-station LSTM and Transformer; the proposed **GAT-Transformer**:
graph attention across stations at every hour → temporal Transformer over 24 h → a second graph-attention
step → direct multi-horizon head (+1, +3, +6, +12, +24 h). *Robust* variants are trained with random
station dropout so they keep working when sensors fail.

**AQI categories** follow the Indian National AQI PM2.5 breakpoints (Good ≤30, Satisfactory ≤60,
Moderate ≤90, Poor ≤120, Very Poor ≤250, Severe >250 µg/m³).
""")
