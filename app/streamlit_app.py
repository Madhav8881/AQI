"""
Delhi PM2.5 Forecaster — interactive demo.

Run from the project root:
    streamlit run app/streamlit_app.py
"""
import json
import os
import sys
from datetime import datetime, time as dtime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import config as C  # noqa: E402
from data import AQIData  # noqa: E402
from models import MODEL_CLASSES, Persistence, SeasonalNaive, build_model, slug  # noqa: E402

st.set_page_config(page_title="Delhi PM2.5 Forecaster", page_icon="🌫️", layout="wide")

MODEL_COLORS = {
    "Persistence": "#9CA3AF", "Seasonal naive": "#C4B5FD", "LSTM": "#3B82F6",
    "Transformer": "#F59E0B", "GAT-Transformer": "#10B981",
    "Robust LSTM": "#1E3A8A", "Robust GAT-Transformer": "#047857",
}
NEURAL = list(MODEL_CLASSES)
BAND_COLORS = {name: color for _, _, name, color in C.AQI_BANDS}


# ----------------------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading data and models…")
def load_all():
    data = AQIData()
    best_path = os.path.join(C.RESULTS_DIR, "best_seeds.json")
    best = json.load(open(best_path)) if os.path.exists(best_path) else {}
    models = {}
    for name in NEURAL:
        seed = best.get(name, "0")
        path = os.path.join(C.CHECKPOINTS_DIR, f"{slug(name)}_seed{seed}.pt")
        if os.path.exists(path):
            m = build_model(name, data.n_nodes, len(C.HORIZONS), window=C.INPUT_WINDOW)
            m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            models[name] = m.eval()
    return data, models


@st.cache_data
def load_results():
    def rd(name):
        p = os.path.join(C.RESULTS_DIR, name)
        return pd.read_csv(p) if os.path.exists(p) else None
    out = {k: rd(f"{k}.csv") for k in
           ["metrics_overall", "metrics_horizon", "metrics_season", "metrics_station", "dm_test"]}
    p = os.path.join(C.RESULTS_DIR, "test_predictions.npz")
    if os.path.exists(p):
        z = np.load(p)
        out["pred_origins"], out["pred_names"], out["preds"] = z["origins"], list(z["names"]), z["preds"]
    logs = {}
    log_dir = os.path.join(C.RESULTS_DIR, "train_logs")
    if os.path.isdir(log_dir):
        for fn in sorted(os.listdir(log_dir)):
            with open(os.path.join(log_dir, fn)) as f:
                logs[fn[:-5]] = json.load(f)
    out["logs"] = logs
    return out


data, models = load_all()
res = load_results()
stations = data.stations
names = stations["name"].tolist()
T = len(data.times)
t_index = pd.Series(np.arange(T), index=data.times)
TRAIN_END, VAL_END = pd.Timestamp(C.TRAIN_END), pd.Timestamp(C.VAL_END)


def period_of(ts):
    if ts <= TRAIN_END:
        return "Training period", "the models saw this data during training"
    if ts <= VAL_END:
        return "Validation period", "used only for early stopping"
    return "Test period", "never seen by the models — a genuine out-of-sample forecast"


def aqi_color(v):
    return BAND_COLORS[C.aqi_category(v)] if np.isfinite(v) else "#D1D5DB"


@torch.no_grad()
def forecast_at(t, hide=None):
    """Run every model on the 24h window ending at hour t. Returns {model: [H, N]}, attention.
    hide: station index whose sensor is simulated as failed (its readings are hidden)."""
    X = data.features[t - C.INPUT_WINDOW + 1: t + 1].unsqueeze(0)
    if hide is not None:
        X = data.hide_stations(X, [hide])
    out = {"Persistence": Persistence(C.HORIZONS)(X), "Seasonal naive": SeasonalNaive(C.HORIZONS)(X)}
    attn = {}
    for name, m in models.items():
        if "GAT" in name:
            out[name], a = m(X, data.adj, return_attn=True)
            attn[name] = a[0].numpy()
        else:
            out[name] = m(X, data.adj)
    out = {k: data.denorm(v.numpy()[0]) for k, v in out.items()}
    return out, attn


def add_aqi_bands(fig, ymax):
    for lo, hi, name, color in C.AQI_BANDS:
        if lo > ymax:
            break
        fig.add_hrect(y0=lo, y1=min(hi, ymax), fillcolor=color, opacity=0.08, line_width=0,
                      annotation_text=name, annotation_position="right",
                      annotation_font_size=10, annotation_font_color="#6B7280")


# ----------------------------------------------------------------------------- sidebar
# A click on the map selects a station (handled before the selectbox is drawn).
ev = st.session_state.get("station_map")
if ev and ev.get("selection", {}).get("points"):
    picked = ev["selection"]["points"][0]["customdata"][0]
    if picked != st.session_state.get("_last_map_pick"):
        st.session_state["_last_map_pick"] = picked
        st.session_state["station"] = picked
st.session_state.setdefault("station", "Anand Vihar")

with st.sidebar:
    st.title("🌫️ Delhi PM2.5 Forecaster")
    st.caption(f"{len(stations)} CPCB/DPCC/IMD monitoring stations · real hourly data "
               f"{data.times[0]:%b %Y} – {data.times[-1]:%b %Y}")
    station = st.selectbox("Station", names, key="station")
    sidx = names.index(station)
    st.markdown("**Forecast moment**")
    d = st.date_input("Date", value=datetime(2024, 11, 18).date(),
                      min_value=data.times[C.INPUT_WINDOW].date(), max_value=data.times[-1].date())
    hr = st.slider("Hour (IST)", 0, 23, 8)
    failed = st.toggle("🔌 Simulate sensor failure", key="failed",
                       help="Hide this station's last 24 hours of readings from every model, as if "
                            "its sensor were broken. Models that only look at the station itself are "
                            "left guessing; the GAT models can still use neighbouring stations.")
    default_models = (["LSTM", "GAT-Transformer", "Robust GAT-Transformer"] if failed
                      else ["Persistence", "LSTM", "GAT-Transformer"])
    available = [m for m in MODEL_COLORS if m in ("Persistence", "Seasonal naive") or m in models]
    # Keyed on `failed` so the selection resets to sensible defaults when the toggle flips.
    shown = st.multiselect("Models to show", available, key=f"shown_{failed}",
                           default=[m for m in default_models if m in available])
    st.divider()
    st.caption("Tip: click a station on the map to select it.")
    if not models:
        st.error("No trained checkpoints found — run `python src/train_eval.py` first.")

ts = pd.Timestamp(datetime.combine(d, dtime(hr)))
t = int(t_index[ts])
window_obs = data.observed[t - C.INPUT_WINDOW + 1: t + 1].mean()

tab_live, tab_bt, tab_cmp, tab_exp, tab_about = st.tabs(
    ["🗺️ Live map & forecast", "📈 Backtest", "🏆 Model comparison", "🔎 Station explorer", "ℹ️ About"])

# ----------------------------------------------------------------------------- live
with tab_live:
    per, why = period_of(ts)
    badge = {"Test period": "🟢", "Validation period": "🟡", "Training period": "🟠"}[per]
    st.markdown(f"#### {station} — forecast issued at {ts:%d %b %Y, %H:00}  \n"
                f"{badge} **{per}**: {why}.")

    if window_obs < 0.5:
        st.warning("The monitoring network reported almost no data in the 24 hours before this "
                   "moment (the source has a network-wide gap in Jan–Mar 2025). Pick another date.")
    else:
        fc, attn_all = forecast_at(t, sidx if failed else None)
        attn_model = "Robust GAT-Transformer" if failed and "Robust GAT-Transformer" in attn_all             else "GAT-Transformer"
        attn = attn_all.get(attn_model)
        if failed:
            st.error(f"🔌 **Sensor failure simulated at {station}.** None of the models can see this "
                     f"station's readings from the last 24 hours. The real values are still drawn "
                     f"(grey) so you can judge who guessed best. With no data of its own, "
                     f"*Persistence* falls back to the station's long-term average.")
        now = data.pm25[t]
        last24 = np.nanmean(data.pm25[t - 23: t + 1], axis=0)

        col_map, col_card = st.columns([3, 2], gap="large")
        with col_map:
            map_metric = st.radio("Colour stations by", ["Observed now", "GAT-Transformer +6h forecast",
                                                         "GAT-Transformer +24h forecast"],
                                  horizontal=True, label_visibility="collapsed")
            if map_metric == "Observed now":
                vals = now
            else:
                h = 6 if "+6h" in map_metric else 24
                src = fc.get("GAT-Transformer", fc["Persistence"])
                vals = src[C.HORIZONS.index(h)]
            mdf = stations.assign(pm25=np.round(vals, 1),
                                  category=[C.aqi_category(v) if np.isfinite(v) else "No data" for v in vals],
                                  selected=[n == station for n in names])
            mdf["size"] = np.where(mdf["selected"], 22, 13)
            fig = px.scatter_map(
                mdf, lat="lat", lon="lon", color="category", size="size", size_max=22,
                color_discrete_map={**BAND_COLORS, "No data": "#D1D5DB"},
                category_orders={"category": [b[2] for b in C.AQI_BANDS] + ["No data"]},
                custom_data=["name", "pm25"], hover_name="name",
                hover_data={"pm25": True, "category": True, "lat": False, "lon": False, "size": False},
                zoom=9.4, center={"lat": 28.65, "lon": 77.17}, height=520, map_style="carto-positron")
            sel = mdf[mdf["selected"]].iloc[0]
            fig.add_trace(go.Scattermap(lat=[sel.lat], lon=[sel.lon], mode="markers",
                                        marker=dict(size=30, color="rgba(0,0,0,0)"),
                                        hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scattermap(lat=[sel.lat], lon=[sel.lon], mode="text", text=["▼ " + station],
                                        textposition="top center", textfont=dict(size=13, color="#111827"),
                                        hoverinfo="skip", showlegend=False))
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend_title_text="AQI category",
                              legend=dict(orientation="h", y=-0.02))
            st.plotly_chart(fig, width="stretch", key="station_map", on_select="rerun",
                            selection_mode="points")

        with col_card:
            cur = now[sidx]
            cat = C.aqi_category(cur) if np.isfinite(cur) else "No data"
            st.markdown(
                f"<div style='border-radius:12px;padding:16px 18px;background:{aqi_color(cur)}22;"
                f"border-left:6px solid {aqi_color(cur)}'>"
                f"<div style='font-size:13px;color:#6B7280'>Observed PM2.5 at {ts:%H:00}"
                f"{' — hidden from models' if failed else ''}</div>"
                f"<div style='font-size:40px;font-weight:700'>"
                f"{'—' if not np.isfinite(cur) else f'{cur:.0f}'}"
                f" <span style='font-size:16px;font-weight:400'>µg/m³</span></div>"
                f"<div style='font-size:16px;font-weight:600'>{cat}</div>"
                f"<div style='font-size:13px;color:#6B7280'>24-hour average: {last24[sidx]:.0f} µg/m³ · "
                f"city rank {int((np.nan_to_num(now, nan=-1) > np.nan_to_num(cur, nan=-1)).sum()) + 1}"
                f" of {len(names)}</div></div>", unsafe_allow_html=True)

            rows = []
            for hi, h in enumerate(C.HORIZONS):
                tt = t + h
                actual = data.pm25[tt, sidx] if tt < T else np.nan
                row = {"Lead": f"+{h}h", "Time": data.times[tt].strftime("%d %b %H:00") if tt < T else "—"}
                for name in shown:
                    if name in fc:
                        row[name] = round(float(fc[name][hi, sidx]), 0)
                row["Actual"] = None if not np.isfinite(actual) else round(float(actual), 0)
                rows.append(row)
            st.markdown("**Forecasts (µg/m³)**")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        # Forecast chart: 48h history + forecasts + what actually happened.
        lo, hi_t = max(0, t - 47), min(T - 1, t + max(C.HORIZONS))
        hist_t = data.times[lo: t + 1]
        fut_t = data.times[t + 1: hi_t + 1]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_t, y=data.pm25[lo: t + 1, sidx],
                                 name="Observed (hidden from models)" if failed else "Observed",
                                 line=dict(color="#9CA3AF" if failed else "#111827", width=2.5)))
        fig.add_trace(go.Scatter(x=fut_t, y=data.pm25[t + 1: hi_t + 1, sidx], name="Actual (what happened)",
                                 line=dict(color="#111827", width=2, dash="dot")))
        for name in shown:
            if name not in fc:
                continue
            xs = [data.times[t] + pd.Timedelta(hours=h) for h in C.HORIZONS]
            ys = list(fc[name][:, sidx])
            if not failed and np.isfinite(data.pm25[t, sidx]):   # connect to the last observation
                xs, ys = [data.times[t]] + xs, [data.pm25[t, sidx]] + ys
            fig.add_trace(go.Scatter(x=xs, y=ys, name=name, mode="lines+markers",
                                     line=dict(color=MODEL_COLORS[name], width=2.5 if "GAT" in name else 1.8),
                                     marker=dict(size=7)))
        ymax = np.nanmax([np.nanmax(data.pm25[lo: hi_t + 1, sidx])] +
                         [fc[n][:, sidx].max() for n in shown if n in fc]) * 1.15
        add_aqi_bands(fig, ymax)
        fig.add_vline(x=data.times[t], line_dash="dash", line_color="#6B7280")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="PM2.5 (µg/m³)",
                          yaxis_range=[0, ymax], hovermode="x unified",
                          title=dict(text="Past 48 h and next 24 h", font_size=15),
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, width="stretch")

        if attn is not None:
            st.markdown(f"**Which neighbouring stations did the {attn_model} listen to?**  "
                        "Graph-attention weights for this station at this moment "
                        "(they change with the pollution situation).")
            w = attn[sidx]
            nb = np.where(w > 0.005)[0]
            adf = pd.DataFrame({"Station": [names[j] + (" (itself)" if j == sidx else "") for j in nb],
                                "Attention": w[nb], "Distance (km)": data.dist[sidx, nb].round(1),
                                "PM2.5 now": np.round(now[nb], 0)}).sort_values("Attention")
            fig = px.bar(adf, x="Attention", y="Station", orientation="h",
                         hover_data=["Distance (km)", "PM2.5 now"], color_discrete_sequence=["#10B981"])
            fig.update_layout(height=max(220, 28 * len(adf)), margin=dict(l=10, r=10, t=10, b=10),
                              xaxis_tickformat=".0%")
            st.plotly_chart(fig, width="stretch")

# ----------------------------------------------------------------------------- backtest
with tab_bt:
    if "preds" not in res:
        st.info("Run the evaluation (`python src/train_eval.py --eval-only`) to enable backtesting.")
    else:
        po = res["pred_origins"]
        pt = data.times[po]
        st.markdown(f"#### {station}: forecasts vs reality over the unseen test period")
        c1, c2, c3 = st.columns([2, 1, 2])
        with c1:
            rng = st.date_input("Date range", value=(datetime(2024, 11, 1).date(),
                                                      datetime(2024, 11, 30).date()),
                                min_value=pt[0].date(), max_value=pt[-1].date())
        with c2:
            h = st.selectbox("Lead time", C.HORIZONS, index=2, format_func=lambda x: f"+{x} h")
        with c3:
            bt_models = st.multiselect("Models", res["pred_names"],
                                       default=[m for m in ["Persistence", "LSTM", "GAT-Transformer"]
                                                if m in res["pred_names"]], key="bt_models")
        if isinstance(rng, tuple) and len(rng) == 2:
            hi_ = C.HORIZONS.index(h)
            valid_t = pt + pd.Timedelta(hours=h)
            sel = (valid_t >= pd.Timestamp(rng[0])) & (valid_t < pd.Timestamp(rng[1]) + pd.Timedelta(days=1))
            if sel.sum() == 0:
                st.warning("No test forecasts in this range (Jan–Mar 2025 has no source data).")
            else:
                actual = data.pm25[po[sel] + h, sidx]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=valid_t[sel], y=actual, name="Actual",
                                         line=dict(color="#111827", width=2)))
                mrows = []
                for name in bt_models:
                    p = res["preds"][res["pred_names"].index(name)][sel, hi_, sidx]
                    fig.add_trace(go.Scatter(x=valid_t[sel], y=p, name=name,
                                             line=dict(color=MODEL_COLORS[name], width=1.6)))
                    ok = np.isfinite(actual)
                    err = p[ok] - actual[ok]
                    mrows.append({"Model": name, "MAE": np.abs(err).mean(),
                                  "RMSE": np.sqrt((err ** 2).mean()), "Bias": err.mean()})
                ymax = np.nanmax(actual) * 1.15
                add_aqi_bands(fig, ymax)
                fig.update_layout(height=460, yaxis_title="PM2.5 (µg/m³)", hovermode="x unified",
                                  yaxis_range=[0, ymax], margin=dict(l=10, r=10, t=30, b=10),
                                  title=dict(text=f"+{h} h ahead forecasts", font_size=15),
                                  legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig, width="stretch")
                st.markdown("**Error for this station and date range (µg/m³)**")
                st.dataframe(pd.DataFrame(mrows).round(2).sort_values("MAE"), hide_index=True)

# ----------------------------------------------------------------------------- comparison
with tab_cmp:
    ov = res.get("metrics_overall")
    if ov is None:
        st.info("Run the evaluation to see model comparison.")
    else:
        st.markdown("#### Test-set accuracy — all 36 stations, real observations only")
        st.caption("Test period: Oct–Dec 2024 (winter smog) and Apr–Aug 2025 (summer/monsoon). "
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
            fig.update_layout(title="MAE by season / episode", yaxis_title="MAE (µg/m³)",
                              xaxis_title=None, height=380, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, width="stretch")

        sf_path = os.path.join(C.RESULTS_DIR, "sensor_failure.csv")
        if os.path.exists(sf_path):
            sf = pd.read_csv(sf_path)
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
                                 center={"lat": 28.65, "lon": 77.17}, height=420,
                                 map_style="carto-positron")
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
            fig = px.line(pd.DataFrame(rows), x="epoch", y="loss", color="run", line_dash="split",
                          height=380)
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch")

# ----------------------------------------------------------------------------- explorer
with tab_exp:
    s = pd.Series(data.pm25[:, sidx], index=data.times)
    city = pd.DataFrame(data.pm25, index=data.times).mean(axis=1)
    info = stations.iloc[sidx]
    st.markdown(f"#### {station} ({info['agency']}) — {info['lat']:.4f}°N, {info['lon']:.4f}°E")
    k = st.columns(4)
    k[0].metric("Average PM2.5 (2023–25)", f"{s.mean():.0f} µg/m³",
                f"{s.mean() - city.mean():+.0f} vs city", delta_color="inverse")
    k[1].metric("Worst hour", f"{s.max():.0f} µg/m³", s.idxmax().strftime("%d %b %Y %H:00"),
                delta_color="off")
    k[2].metric("Hours ‘Very Poor’ or worse", f"{(s >= 120).mean():.0%}")
    k[3].metric("Real-data coverage", f"{info['coverage']:.0%}")

    daily = pd.DataFrame({station: s.resample("D").mean(), "Delhi average": city.resample("D").mean()})
    fig = px.line(daily, height=380, color_discrete_sequence=["#10B981", "#9CA3AF"])
    fig.update_layout(yaxis_title="Daily mean PM2.5 (µg/m³)", xaxis_title=None, legend_title=None,
                      margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified")
    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        monthly = s.groupby(s.index.month).mean()
        fig = px.bar(x=[pd.Timestamp(2000, m, 1).strftime("%b") for m in monthly.index], y=monthly.values,
                     color=[C.aqi_category(v) for v in monthly.values], color_discrete_map=BAND_COLORS)
        fig.update_layout(title="Average by month", yaxis_title="PM2.5 (µg/m³)", xaxis_title=None,
                          legend_title=None, height=340, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")
    with c2:
        season = np.where(s.index.month.isin([11, 12, 1, 2]), "Winter (Nov–Feb)",
                          np.where(s.index.month.isin([7, 8, 9]), "Monsoon (Jul–Sep)", "Other months"))
        diurnal = s.groupby([season, s.index.hour]).mean().rename_axis(["Season", "Hour"]).reset_index(name="PM2.5")
        fig = px.line(diurnal, x="Hour", y="PM2.5", color="Season", markers=True,
                      color_discrete_sequence=["#6B7280", "#3B82F6", "#EF4444"])
        fig.update_layout(title="Daily cycle (hour of day)", yaxis_title="PM2.5 (µg/m³)", height=340,
                          margin=dict(l=10, r=10, t=40, b=10), legend_title=None)
        st.plotly_chart(fig, width="stretch")

# ----------------------------------------------------------------------------- about
with tab_about:
    st.markdown(f"""
#### How it works
**Data.** Real hourly PM2.5 from {len(stations)} Delhi monitoring stations (CPCB, DPCC, IMD) via the
XKDR Air Quality API, Jan 2023 – Aug 2025. No synthetic data. Stuck sensors (≥12 identical hours) and
non-physical values are removed. Jan–Mar 2025 is missing network-wide in the source and is skipped.

**Split (chronological).** Train Jan 2023 – Jun 2024 · Validation Jul – Sep 2024 ·
Test Oct – Dec 2024 + Apr – Aug 2025.

**No leakage.** Scalers are fitted on training data only; missing inputs are filled *causally*
(forward-fill only) with an `observed` flag as an extra input; models are trained and scored only on
real observed targets.

**Graph.** Each station connects to its {C.KNN} nearest stations; edges weighted by a Gaussian kernel of
Haversine distance. The GAT uses log(weight) as an attention prior that it can learn to override.

**Models.** Persistence and seasonal-naive baselines; per-station LSTM and Transformer; the proposed
**GAT-Transformer**: graph attention across stations at every hour → temporal Transformer over 24 h →
a second graph-attention step → direct multi-horizon head (+1, +3, +6, +12, +24 h). All neural models
share station embeddings and residual forecasting so the comparison is fair.

**AQI categories** follow the Indian National AQI PM2.5 breakpoints (Good ≤30, Satisfactory ≤60,
Moderate ≤90, Poor ≤120, Very Poor ≤250, Severe >250 µg/m³).
""")
