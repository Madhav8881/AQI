"""
Delhi Air Quality Forecast — live demo, written for everyone (no technical background needed).

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
# "How good is it?" tab always shows the evaluated (hold-out) models' test results.
if "AQI_RUN" not in os.environ and os.path.exists(os.path.join(ROOT, "results", "final", "best_seeds.json")):
    os.environ["AQI_RUN"] = "final"

import config as C  # noqa: E402
from data import AQIData  # noqa: E402
from models import MODEL_CLASSES, Persistence, build_model, slug  # noqa: E402
import live as L  # noqa: E402

st.set_page_config(page_title="Delhi Air Quality Forecast", page_icon="🌫️", layout="wide")

MODEL_COLORS = {
    "Persistence": "#9CA3AF", "Seasonal naive": "#C4B5FD", "LSTM": "#3B82F6",
    "Transformer": "#F59E0B", "GAT-Transformer": "#10B981",
    "Robust LSTM": "#1E3A8A", "Robust GAT-Transformer": "#047857", "Ensemble": "#7C3AED",
}
# Plain-language names first, the technical name in brackets for those who want it.
FRIENDLY = {
    "Persistence": "Simple guess: no change (Persistence)",
    "Seasonal naive": "Same hour yesterday",
    "LSTM": "Standard model (LSTM)",
    "Transformer": "Standard model (Transformer)",
    "GAT-Transformer": "Our model (GAT-Transformer)",
    "Robust LSTM": "Standard model, sensor-proof (Robust LSTM)",
    "Robust GAT-Transformer": "Our model, sensor-proof (Robust GAT)",
    "Ensemble": "Our combined forecast (Ensemble)",
}


def nm(name):
    return FRIENDLY.get(name, name)


COLORS_BY_NAME = {nm(k): v for k, v in MODEL_COLORS.items()}
BAND_COLORS = {name: color for _, _, name, color in C.AQI_BANDS}
ADVICE = {
    "Good": "Clean air. A great time to be outdoors.",
    "Satisfactory": "Fine for most people. Very sensitive people may notice slight discomfort.",
    "Moderate": "People with asthma, heart or lung conditions, children and older adults should avoid "
                "long or hard exercise outdoors.",
    "Poor": "Most people may feel discomfort if outdoors for long. Cut back on heavy outdoor exercise; "
            "sensitive people should stay indoors more.",
    "Very Poor": "Avoid outdoor exercise. Sensitive people should stay indoors. Consider a well-fitted "
                 "N95 mask if you go out.",
    "Severe": "Avoid going outside if you can. Keep windows closed, use an air purifier if you have one, "
              "and wear an N95 mask if you must go out.",
}


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


@st.cache_resource(show_spinner="Loading the forecasting models…")
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


# ----------------------------------------------------------------------------- helpers
def aqi_color(v):
    return BAND_COLORS[C.aqi_category(v)] if np.isfinite(v) else "#D1D5DB"


def add_aqi_bands(fig, ymax):
    for lo, hi, name, color in C.AQI_BANDS:
        if lo > ymax:
            break
        fig.add_hrect(y0=lo, y1=min(hi, ymax), fillcolor=color, opacity=0.08, line_width=0,
                      annotation_text=name, annotation_position="right",
                      annotation_font_size=10, annotation_font_color="#6B7280")


def colour_chip(top, big, cat, small=""):
    """A small coloured card: label, big number, air-quality category, optional extra line."""
    color = BAND_COLORS.get(cat, "#D1D5DB")
    return (f"<div style='border-radius:10px;padding:10px 8px;background:{color}22;"
            f"border-top:5px solid {color};text-align:center;height:100%'>"
            f"<div style='font-size:12px;opacity:.7'>{top}</div>"
            f"<div style='font-size:24px;font-weight:700;line-height:1.25'>{big}</div>"
            f"<div style='font-size:13px;font-weight:600'>{cat}</div>"
            f"<div style='font-size:11px;opacity:.7'>{small}</div></div>")


def clock(t, ref):
    """'6 PM', '6 PM tomorrow' — time of day relative to the reference time."""
    s = f"{t:%I %p}".lstrip("0")
    days = (t.normalize() - ref.normalize()).days
    return s if days == 0 else (f"{s} tomorrow" if days == 1 else f"{s}, {t:%d %b}")


def render_legend():
    cols = st.columns(len(C.AQI_BANDS))
    for col, (lo, hi, name, color) in zip(cols, C.AQI_BANDS):
        rng = f"{lo:g}–{hi:g}" if hi < 1000 else f"over {lo:g}"
        col.markdown(f"<div style='border-radius:8px;padding:6px 4px;background:{color}33;"
                     f"border-top:4px solid {color};text-align:center'>"
                     f"<div style='font-size:13px;font-weight:700'>{name}</div>"
                     f"<div style='font-size:11px;opacity:.75'>{rng} µg/m³</div></div>",
                     unsafe_allow_html=True)


def outlook_text(cur, latest, vals):
    """One or two plain sentences about the next 24 hours from the forecast values."""
    times = [latest + pd.Timedelta(hours=h) for h in C.HORIZONS]
    best, worst = int(np.argmin(vals)), int(np.argmax(vals))
    parts = []
    if np.isfinite(cur):
        d = vals[0] - cur
        parts.append("In the next hour it is expected to "
                     + ("get worse" if d > 15 else "improve" if d < -15 else "stay about the same") + ".")
    if vals[worst] - vals[best] < 20:
        mean = float(np.mean(vals))
        parts.append(f"Over the next 24 hours the air should stay fairly steady, around {mean:.0f} µg/m³ "
                     f"({C.aqi_category(mean)}).")
    else:
        parts.append(f"Cleanest expected around **{clock(times[best], latest)}** "
                     f"({vals[best]:.0f}, {C.aqi_category(vals[best])}); worst around "
                     f"**{clock(times[worst], latest)}** ({vals[worst]:.0f}, {C.aqi_category(vals[worst])}).")
    return " ".join(parts)


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
    st.title("🌫️ Delhi Air Quality Forecast")
    st.caption("Shows how polluted Delhi's air is right now, and forecasts the next 24 hours. "
               "It tracks **PM2.5** — tiny particles in the air that harm lungs and hearts.")
    station = st.selectbox("Choose a station", names, key="station")
    sidx = names.index(station)
    st.caption("Tip: you can also click a dot on the map.")
    st.divider()
    failed = st.toggle("🔌 Pretend this station's sensor is broken", key="failed",
                       help="Hides this station's own readings from the forecasting models, as if the "
                            "sensor had stopped working. It shows how well the models can still work "
                            "out the air quality from the stations around it.")
    st.divider()
    st.caption(f"Forecasts come from a deep-learning model trained on {len(stations)} Delhi stations' "
               "hourly readings from Jan 2023 to Sep 2026. Live readings: "
               "[World Air Quality Index project](https://aqicn.org), which republishes the official "
               "CPCB / DPCC stations.")
    if not models:
        st.error("No trained models found — run `python src/train_eval.py` first.")

tab_today, tab_real, tab_cmp, tab_about = st.tabs(
    ["📡 Today", "🎯 Prediction vs reality", "🏆 How good is it?", "ℹ️ About"])


# ----------------------------------------------------------------------------- today
def render_today():
    token = L.load_token()
    if not token:
        st.info("Live data needs a free token from aqicn.org/data-platform/token. Add it as "
                "`WAQI_TOKEN=...` in the `.env` file in the project folder.")
        return

    top, btn = st.columns([5, 1])
    stale = (not os.path.exists(L.READINGS_CSV) or
             pd.Timestamp.now().timestamp() - os.path.getmtime(L.READINGS_CSV) > 20 * 60)
    if btn.button("🔄 Refresh", width="stretch") or stale:
        with st.spinner("Getting the latest readings from Delhi's stations…"):
            try:
                L.collect(data, models, token)
            except Exception as e:
                st.warning(f"Could not get new readings right now ({e}). Showing the latest saved ones.")
    readings, fcs = L.load_readings(), L.load_forecasts()
    if readings.empty:
        st.warning("No readings yet — press “Refresh”.")
        return

    latest = readings["time"].max()
    recent = readings[readings["time"] >= latest - pd.Timedelta(hours=2)]
    now_vals = stations["station_id"].map(recent.sort_values("time").groupby("station_id")["pm25"].last()
                                          ).values.astype(float)
    ok = np.isfinite(now_vals)
    hours_have = readings.loc[readings["time"] > latest - pd.Timedelta(hours=C.INPUT_WINDOW), "time"].nunique()

    top.markdown(f"### Delhi air quality right now\n{latest:%A, %d %B %Y · %I:%M %p} (Indian time)")
    with st.expander("What do the colours and numbers mean?"):
        st.markdown("**PM2.5** is the amount of tiny particles (smaller than 2.5 micrometres) in the air, from "
                    "traffic, dust, burning and industry. They get deep into the lungs. It is measured in "
                    "**µg/m³** (micrograms per cubic metre): a higher number means dirtier air.")
        render_legend()
        st.caption("Colours follow India's national air quality categories for PM2.5. The advice shown is "
                   "general guidance, not medical advice.")

    # City-wide summary.
    avg = float(np.nanmean(now_vals))
    lo_i, hi_i = int(np.nanargmin(now_vals)), int(np.nanargmax(now_vals))
    k = st.columns(4)
    k[0].markdown(colour_chip("Delhi average", f"{avg:.0f}", C.aqi_category(avg), "µg/m³"), unsafe_allow_html=True)
    k[1].markdown(colour_chip("Most polluted", f"{now_vals[hi_i]:.0f}", C.aqi_category(now_vals[hi_i]),
                              names[hi_i]), unsafe_allow_html=True)
    k[2].markdown(colour_chip("Cleanest", f"{now_vals[lo_i]:.0f}", C.aqi_category(now_vals[lo_i]),
                              names[lo_i]), unsafe_allow_html=True)
    k[3].metric("Stations reporting", f"{int(ok.sum())} of {len(stations)}")
    if hours_have < C.INPUT_WINDOW:
        st.caption(f"ℹ️ Only {hours_have} hour(s) of readings have been saved so far, so forecasts are slightly "
                   f"less sharp than usual. They sharpen automatically as more readings are saved.")

    fc, attn = live_forecast(readings, latest, hide=sidx if failed else None)
    primary = next((m for m in (["Robust GAT-Transformer", "GAT-Transformer"] if failed
                                else ["Ensemble", "GAT-Transformer"]) if m in fc), "Persistence")
    if failed:
        st.warning(f"🔌 **Pretending {station}'s sensor is broken.** The models can't see this station's own "
                   f"readings, so they have to work it out from the stations around it. The grey line below "
                   f"is what the sensor really reported.")

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
            labels={"pm25": "PM2.5 (µg/m³)", "category": "Air quality"},
            zoom=9.4, center={"lat": 28.65, "lon": 77.17}, height=480, map_style="carto-positron")
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend_title_text="Air quality",
                          legend=dict(orientation="h", y=-0.02))
        st.plotly_chart(fig, width="stretch", key="today_map", on_select="rerun", selection_mode="points")

    cur = now_vals[sidx]
    with col_c:
        cat = C.aqi_category(cur) if np.isfinite(cur) else None
        color = aqi_color(cur)
        label = f"{station} — actual reading, hidden from the models" if failed else f"{station} — right now"
        st.markdown(
            f"<div style='border-radius:12px;padding:16px 18px;background:{color}22;"
            f"border-left:6px solid {color}'>"
            f"<div style='font-size:13px;opacity:.7'>{label}</div>"
            f"<div style='font-size:42px;font-weight:700;line-height:1.2'>"
            f"{'—' if not np.isfinite(cur) else f'{cur:.0f}'}"
            f" <span style='font-size:16px;font-weight:400'>µg/m³</span></div>"
            f"<div style='font-size:18px;font-weight:600'>{cat or 'Station offline right now'}</div>"
            + (f"<div style='font-size:14px;margin-top:6px'>{ADVICE[cat]}</div>" if cat else "")
            + "</div>", unsafe_allow_html=True)
        pv = fc[primary][:, sidx]
        st.markdown("**What to expect**")
        st.markdown(outlook_text(cur, latest, pv))
        if not np.isfinite(cur) and not failed:
            st.caption("This station isn't reporting right now, so its forecast is worked out from the "
                       "stations around it.")

    # The next 24 hours at a glance.
    st.markdown(f"#### {station}: the next 24 hours")
    cols = st.columns(len(C.HORIZONS))
    for col, h, v in zip(cols, C.HORIZONS, pv):
        t = latest + pd.Timedelta(hours=h)
        col.markdown(colour_chip(clock(t, latest), f"{v:.0f}", C.aqi_category(v), "µg/m³"), unsafe_allow_html=True)
    hz = res.get("metrics_horizon")
    if hz is not None and not failed:
        g = hz[hz.Model == "GAT-Transformer"].set_index("Horizon")["MAE_mean"]
        if 1 in g.index and 24 in g.index:
            st.caption(f"How much to trust it: over two years of past tests, forecasts were off by about "
                       f"{g[1]:.0f} µg/m³ one hour ahead and about {g[24]:.0f} µg/m³ a day ahead, on average. "
                       f"Sudden changes (a gust of wind, rain, fires) can't be predicted.")

    # Chart: what was measured, and what we expect.
    st_read = readings[readings["station_id"] == stations["station_id"].iloc[sidx]].sort_values("time")
    show = [primary, "Persistence"] + (["LSTM"] if failed and "LSTM" in fc else [])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=st_read["time"], y=st_read["pm25"], mode="lines+markers",
                             name="What the sensor measured" + (" (hidden from models)" if failed else ""),
                             line=dict(color="#9CA3AF" if failed else "#111827", width=2.5)))
    for name in show:
        fig.add_trace(go.Scatter(x=[latest + pd.Timedelta(hours=h) for h in C.HORIZONS],
                                 y=fc[name][:, sidx], name=nm(name), mode="lines+markers",
                                 line=dict(color=MODEL_COLORS.get(name), width=3 if name == primary else 1.8)))
    yvals = list(st_read["pm25"].dropna()) + [float(fc[n][:, sidx].max()) for n in show]
    ymax = max(yvals) * 1.15 if yvals else 300
    add_aqi_bands(fig, ymax)
    fig.add_vline(x=latest, line_dash="dash", line_color="#6B7280", annotation_text="now")
    fig.update_layout(height=400, yaxis_title="PM2.5 (µg/m³) — higher is dirtier", yaxis_range=[0, ymax],
                      hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.14),
                      title=dict(text="What was measured, and what we expect", font_size=15))
    st.plotly_chart(fig, width="stretch")

    # Live sensor-failure check: estimate the current hour without the station's own data.
    if failed:
        prev_end = latest - pd.Timedelta(hours=1)
        if not np.isfinite(cur) or readings["time"].min() > prev_end:
            st.caption("The live check needs this station's current reading and at least one earlier hour "
                       "of saved readings.")
        else:
            fc_prev, _ = live_forecast(readings[readings["time"] <= prev_end], prev_end, hide=sidx)
            chk = pd.DataFrame([{"Model": nm(n), "Its estimate for now (µg/m³)": round(float(fc_prev[n][0, sidx])),
                                 "How far off (µg/m³)": round(abs(float(fc_prev[n][0, sidx]) - cur))}
                                for n in fc_prev if n in ("Persistence", "LSTM", "Ensemble")
                                or "GAT" in n]).sort_values("How far off (µg/m³)")
            st.markdown(f"**Who copes best without the sensor?**  Imagine {station}'s sensor stopped an hour ago. "
                        f"Each model estimates the value for {latest:%I:%M %p} using only the surrounding "
                        f"stations. The real reading is **{cur:.0f} µg/m³**; the smaller the gap, the better.")
            st.dataframe(chk, hide_index=True, width="stretch")

    attn_model = "Robust GAT-Transformer" if failed and "Robust GAT-Transformer" in attn else "GAT-Transformer"
    if attn_model in attn:
        with st.expander(f"Which nearby stations influenced this forecast?"):
            st.markdown(f"Our model looks at the stations around {station} to forecast it. A longer bar "
                        f"means that station mattered more right now.")
            w = attn[attn_model][sidx]
            nb = np.where(w > 0.005)[0]
            adf = pd.DataFrame({"Station": [names[j] + (" (itself)" if j == sidx else "") for j in nb],
                                "Influence": w[nb], "Distance (km)": data.dist[sidx, nb].round(1),
                                "PM2.5 now": np.round(now_vals[nb], 0)}).sort_values("Influence")
            fig = px.bar(adf, x="Influence", y="Station", orientation="h",
                         hover_data=["Distance (km)", "PM2.5 now"], color_discrete_sequence=["#10B981"])
            fig.update_layout(height=max(220, 28 * len(adf)), margin=dict(l=10, r=10, t=10, b=10),
                              xaxis_tickformat=".0%")
            st.plotly_chart(fig, width="stretch")

    # Live scoring.
    st.markdown("#### Has the forecast been right?")
    st.markdown("Every hour the app saves its forecasts. When the real reading for that hour arrives, it checks "
                "how far off the forecast was. **Smaller is better.** “Simple guess” just assumes nothing changes.")
    sc = L.live_scores(readings, fcs)
    if sc.empty:
        st.caption("Nothing to check yet — the first forecast reaches its time (one hour ahead) within an hour.")
    else:
        piv = sc.pivot(index="model", columns="horizon", values="MAE").round(1)
        piv.columns = [f"{h} h ahead (µg/m³ off)" for h in piv.columns]
        piv["forecasts checked"] = sc.groupby("model")["n"].sum()
        piv.index = [nm(m) for m in piv.index]
        piv.index.name = "Model"
        st.dataframe(piv.sort_values(piv.columns[0]), width="stretch")

    with st.expander("Compare every model (advanced)"):
        opts = list(fc)
        chosen = st.multiselect("Models", opts, default=opts, format_func=nm, key=f"adv_{failed}")
        tbl = pd.DataFrame({nm(m): np.round(fc[m][:, sidx]).astype(int) for m in chosen},
                           index=[clock(latest + pd.Timedelta(hours=h), latest) for h in C.HORIZONS])
        tbl.index.name = "When"
        st.dataframe(tbl, width="stretch")


with tab_today:
    render_today()


# ----------------------------------------------------------------------------- prediction vs reality
CAT_EDGES = np.array([b[1] for b in C.AQI_BANDS[:-1]])


def render_reality():
    st.markdown("### Prediction vs reality")
    st.markdown("Every hour the app saves what the model **predicts** for the next 1–24 hours. When that hour "
                "arrives, it records what the sensor **actually measured**. This page puts the two side by side.")
    readings, fcs = L.load_readings(), L.load_forecasts()
    sc = L.scored_forecasts(readings, fcs)
    if sc.empty:
        st.info("No forecast has reached its time yet. The first comparisons appear about an hour after the "
                "collector starts, and the full picture builds up over several days. "
                "(Start it with `scripts/start_collector.bat` and leave the window open.)")
        return
    sc = sc.assign(err=sc["pred"] - sc["actual"])
    sc["abs_err"] = sc["err"].abs()
    sc["band_ok"] = np.digitize(sc["pred"], CAT_EDGES) == np.digitize(sc["actual"], CAT_EDGES)
    primary = "Ensemble" if (sc["model"] == "Ensemble").any() else "GAT-Transformer"
    first, last = sc["issued_at"].min(), sc["target_time"].max()

    k = st.columns(4)
    k[0].metric("Period covered", f"{first:%d %b} – {last:%d %b}")
    k[1].metric("Hours with forecasts", f"{sc['issued_at'].nunique()}")
    k[2].metric("Forecasts checked", f"{int((sc['model'] == primary).sum()):,}")
    k[3].metric("Stations", f"{sc['station_id'].nunique()}")

    g = sc.groupby(["model", "horizon"]).agg(miss=("abs_err", "mean"), bias=("err", "mean"),
                                             band=("band_ok", "mean"), n=("abs_err", "size"))
    leads = sorted(int(h) for h in sc["horizon"].unique())
    lead = st.radio("How far ahead?", leads, index=min(2, len(leads) - 1), horizontal=True, key="real_lead",
                    format_func=lambda h: f"{h} hour{'s' if h > 1 else ''} ahead")
    a = g.loc[(primary, lead)]
    c = st.columns(4)
    c[0].metric("Our forecast misses by", f"{a.miss:.0f} µg/m³",
                help="Average gap between what we predicted and what the sensor measured. Smaller is better.")
    if ("Persistence", lead) in g.index:
        b = g.loc[("Persistence", lead)]
        c[1].metric("Simple guess misses by", f"{b.miss:.0f} µg/m³",
                    help="The simple guess assumes the air stays exactly as it is now.")
        c[2].metric("Ours is closer by", f"{1 - a.miss / b.miss:.0%}")
        c[3].metric("Right colour band", f"{a.band:.0%}", f"{(a.band - b.band) * 100:+.0f} points vs simple guess")
    bias = float(sc.loc[sc["model"] == primary, "err"].mean())
    st.caption(f"Across all forecasts, our predictions were on average {abs(bias):.0f} µg/m³ "
               f"{'higher' if bias > 0 else 'lower'} than reality.")

    sid = stations["station_id"].iloc[sidx]
    st.markdown(f"#### {station}: forecasts made {lead} hour{'s' if lead > 1 else ''} ahead, and what happened")
    d = sc[(sc["station_id"] == sid) & (sc["horizon"] == lead)]
    if d.empty:
        st.caption("No checked forecasts for this station yet.")
    else:
        fig = go.Figure()
        act = d[d["model"] == primary].sort_values("target_time")
        fig.add_trace(go.Scatter(x=act["target_time"], y=act["actual"], name="What the sensor measured",
                                 mode="lines+markers", line=dict(color="#111827", width=2.5)))
        for m in [primary, "Persistence"]:
            x = d[d["model"] == m].sort_values("target_time")
            fig.add_trace(go.Scatter(x=x["target_time"], y=x["pred"], name=nm(m), mode="lines+markers",
                                     line=dict(color=MODEL_COLORS.get(m), width=3 if m == primary else 1.6)))
        ymax = float(max(d["actual"].max(), d["pred"].max())) * 1.12
        add_aqi_bands(fig, ymax)
        fig.update_layout(height=420, yaxis_title="PM2.5 (µg/m³) — higher is dirtier", yaxis_range=[0, ymax],
                          hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10),
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, width="stretch")
        st.caption("Each point is plotted at the time it was forecast FOR. The simple guess always lags behind "
                   "reality, because it only repeats the reading from several hours earlier.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**All stations, {lead} hour{'s' if lead > 1 else ''} ahead**")
        d_all = sc[(sc["model"] == primary) & (sc["horizon"] == lead)]
        pts = d_all.sample(min(len(d_all), 4000), random_state=0)
        lim = float(max(pts["actual"].max(), pts["pred"].max())) * 1.05
        fig = px.scatter(pts, x="actual", y="pred", opacity=0.35, height=400,
                         labels={"actual": "What the sensor measured (µg/m³)", "pred": "What we predicted (µg/m³)"},
                         color_discrete_sequence=[MODEL_COLORS.get(primary, "#7C3AED")])
        fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", name="Perfect forecast",
                                 line=dict(color="#6B7280", dash="dash")))
        fig.update_layout(xaxis_range=[0, lim], yaxis_range=[0, lim], margin=dict(l=10, r=10, t=10, b=10),
                          showlegend=False)
        st.plotly_chart(fig, width="stretch")
        st.caption("Each dot is one forecast. The closer to the dashed line, the better the forecast.")
    with c2:
        st.markdown("**Every forecast time**")
        rows = []
        for h in leads:
            if (primary, h) in g.index and ("Persistence", h) in g.index:
                a_, b_ = g.loc[(primary, h)], g.loc[("Persistence", h)]
                rows.append({"Hours ahead": h, "Our forecast misses by (µg/m³)": round(a_.miss, 1),
                             "Simple guess misses by (µg/m³)": round(b_.miss, 1),
                             "Ours is closer by": f"{1 - a_.miss / b_.miss:.0%}",
                             "Right colour band": f"{a_.band:.0%}", "Forecasts checked": int(a_.n)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    with st.expander("Results for every station"):
        sd = sc[sc["horizon"] == lead].groupby(["station_id", "model"])["abs_err"].mean().unstack("model")
        sd = sd[[m for m in (primary, "Persistence") if m in sd.columns]]
        sd.index = sd.index.map(stations.set_index("station_id")["name"])
        sd.columns = [nm(m) for m in sd.columns]
        sd.index.name = f"Station ({lead} h ahead, average miss in µg/m³)"
        st.dataframe(sd.round(1).sort_values(sd.columns[0]), width="stretch")
    st.download_button("⬇️ Download every checked forecast (CSV)", sc.drop(columns=["abs_err", "band_ok"]).to_csv(index=False),
                       file_name="prediction_vs_reality.csv", mime="text/csv")
    st.caption("Live readings come from the World Air Quality Index project, which converts the official stations' "
               "AQI values to µg/m³; they can differ slightly from CPCB's own files, and readings above about "
               "500 µg/m³ are capped.")


with tab_real:
    render_reality()


# ----------------------------------------------------------------------------- how good is it
def render_quality():
    ov = res.get("metrics_overall")
    if ov is None:
        st.info("The accuracy results aren't available yet — run the evaluation first.")
        return

    best = ov.loc[ov["MAE_mean"].idxmin()]
    pers = ov.loc[ov["Model"] == "Persistence", "MAE_mean"].iloc[0]
    st.markdown("### How accurate are the forecasts?")
    st.markdown("We tested the model on **two full years of Delhi data it had never seen** (Oct 2024 – Sep 2026, "
                "including two winters), across all 36 stations. For every hour we compared its forecast with "
                "what the sensor really measured.")
    k = st.columns(3)
    k[0].metric("Typical miss", f"{best['MAE_mean']:.0f} µg/m³",
                help="Average gap between the forecast and the real reading (called MAE). Smaller is better.")
    k[1].metric("Better than a simple guess by", f"{(1 - best['MAE_mean'] / pers):.0%}",
                help="The simple guess assumes the air stays exactly as it is now.")
    k[2].metric("Right colour band", f"{best['AQI_acc_mean']:.0%}",
                help="How often the forecast lands in the same air-quality category (Good, Moderate, Poor, …) "
                     "as the real reading. Being one band off counts as a miss, so this is a strict score.")
    st.caption("Results shown for our best single model. “Typical miss” covers all forecast times from 1 to 24 "
               "hours ahead; it is smaller for near-term forecasts (see the chart below).")

    show = ov.assign(Model=ov["Model"].map(nm))
    st.dataframe(show[["Model", "MAE", "AQI category acc."]].rename(
        columns={"MAE": "Typical miss (µg/m³)", "AQI category acc.": "Right colour band"}),
        hide_index=True, width="stretch")
    with st.expander("More technical measures (RMSE, R²)"):
        st.dataframe(show[["Model", "RMSE", "R²"]], hide_index=True, width="stretch")
        st.caption("RMSE punishes big misses more than small ones (µg/m³, smaller is better). "
                   "R² is the share of the ups and downs the model explains (1 = perfect).")

    hz = res["metrics_horizon"].assign(Model=lambda d: d["Model"].map(nm))
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for name, g in hz.groupby("Model", sort=False):
            fig.add_trace(go.Scatter(x=[f"{h} h" for h in g["Horizon"]], y=g["MAE_mean"],
                                     error_y=dict(type="data", array=g["MAE_std"]),
                                     name=name, mode="lines+markers", line=dict(color=COLORS_BY_NAME.get(name))))
        fig.update_layout(title="Typical miss by how far ahead we forecast", xaxis_title="Hours ahead",
                          yaxis_title="Typical miss (µg/m³)", height=380, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")
    with c2:
        ss = res["metrics_season"].assign(Model=lambda d: d["Model"].map(nm))
        fig = px.bar(ss, x="Subset", y="MAE_mean", color="Model", barmode="group",
                     error_y="MAE_std", color_discrete_map=COLORS_BY_NAME)
        fig.update_layout(title="Typical miss in different periods", yaxis_title="Typical miss (µg/m³)",
                          xaxis_title=None, height=380, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")
    st.caption("Winter smog is much harder to forecast than summer: pollution changes quickly and sharply, so "
               "misses are larger in winter and during severe episodes.")

    sf = res.get("sensor_failure")
    if sf is not None:
        sf = sf.assign(Model=lambda d: d["Model"].map(nm))
        st.markdown("### What if a station's sensor breaks?")
        st.markdown("Sensors fail in real life. We tested every station by hiding its own readings and asking "
                    "the models to work out its air quality from the stations around it. Models that only look "
                    "at one station at a time are left guessing; our graph-based model can use its neighbours.")
        c1, c2 = st.columns([3, 2])
        with c1:
            fig = go.Figure()
            for name, g in sf.groupby("Model", sort=False):
                fig.add_trace(go.Bar(x=[f"{h} h" for h in g["Horizon"]], y=g["MAE_failed"], name=name,
                                     marker_color=COLORS_BY_NAME.get(name, "#D1D5DB")))
            fig.update_layout(barmode="group", xaxis_title="Hours ahead",
                              yaxis_title="Typical miss with the sensor broken (µg/m³)",
                              height=360, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch")
        with c2:
            tbl = sf.groupby("Model", sort=False)[["MAE_normal", "MAE_failed"]].mean()
            tbl["Gets worse by"] = (tbl["MAE_failed"] / tbl["MAE_normal"] - 1).map(
                lambda v: "—" if not np.isfinite(v) else f"{v:+.0%}")
            st.dataframe(tbl.rename(columns={"MAE_normal": "Miss, sensor working",
                                             "MAE_failed": "Miss, sensor broken"}).round(1),
                         width="stretch")

    stm = res["metrics_station"]
    st.markdown("### Where is it most accurate?")
    models_avail = list(stm["Model"].unique())
    mname = st.selectbox("Model", models_avail, format_func=nm,
                         index=models_avail.index("GAT-Transformer") if "GAT-Transformer" in models_avail else 0)
    sdf = stm[stm["Model"] == mname].merge(stations, on="station_id").rename(columns={"MAE": "Typical miss"})
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = px.scatter_map(sdf, lat="lat", lon="lon", color="Typical miss", size="Typical miss",
                             hover_name="name", color_continuous_scale="Viridis_r", zoom=9.2,
                             center={"lat": 28.65, "lon": 77.17}, height=420, map_style="carto-positron")
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.dataframe(sdf[["name", "Typical miss"]].rename(columns={"name": "Station"})
                     .sort_values("Typical miss").round(1), hide_index=True, width="stretch", height=420)

    dm = res.get("dm_test")
    if dm is not None and len(dm):
        with st.expander("Is our model really better, or just lucky? (statistics)"):
            st.markdown("A statistical test (Diebold–Mariano) compares our model with each other model at each "
                        "forecast time. A **negative number** means our model made smaller errors; a **p-value "
                        "below 0.05** means the difference is very unlikely to be chance.")
            piv = dm.assign(cell=[f"{s:+.2f} (p={p:.3f})" for s, p in zip(dm["DM stat"], dm["p-value"])],
                            Comparison=dm["Comparison"].str.replace("GAT-Transformer vs ", "Our model vs ")
                            ).pivot(index="Comparison", columns="Horizon", values="cell")
            piv.columns = [f"{c} h" for c in piv.columns]
            st.dataframe(piv, width="stretch")
    if res["logs"]:
        with st.expander("Training curves (technical)"):
            rows = []
            for key, lg in res["logs"].items():
                for split in ("train", "val"):
                    for ep, v in enumerate(lg["history"][split], 1):
                        rows.append({"run": key, "split": split, "epoch": ep, "loss": v})
            fig = px.line(pd.DataFrame(rows), x="epoch", y="loss", color="run", line_dash="split", height=380)
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch")


with tab_cmp:
    render_quality()

# ----------------------------------------------------------------------------- about
with tab_about:
    st.markdown("""
### What is this?
A live air-quality dashboard for Delhi. It shows how polluted the air is **right now** at each monitoring
station, and **forecasts the next 24 hours** using a deep-learning model.

### What is PM2.5?
Tiny particles in the air, smaller than 2.5 micrometres. They come from traffic, dust, burning and industry,
and they reach deep into the lungs. The app measures them in µg/m³ — the higher the number, the dirtier the air.

### Where do the numbers come from?
- **Live readings:** the official CPCB / DPCC monitoring stations, through the World Air Quality Index project.
- **What the model learned from:** about 3½ years of hourly readings (Jan 2023 – Sep 2026) from 36 Delhi stations.

### How does the forecast work?
The model looks at the **last day of readings** at all stations, plus the time of day, day of the week and time
of year. It pays attention to **neighbouring stations**, because pollution drifts across the city. From that it
predicts the next 1, 3, 6, 12 and 24 hours at every station. It does **not** use weather forecasts.

### How accurate is it?
About **25% closer to reality than simply assuming nothing changes**, tested on two full years it had never seen
(see the “How good is it?” tab). Forecasts are best for the next few hours and get less certain a day ahead.

### What it can't do
- Predict sudden changes such as a gust of wind, rain, a fire or Diwali fireworks before they show up in the readings.
- Replace official health advisories. The advice shown is general guidance, not medical advice.
""")
    with st.expander("Technical details (for the curious)"):
        st.markdown(f"""
**Models.** Persistence baseline; per-station LSTM and Transformer; the proposed **GAT-Transformer**
(graph attention across stations at every hour → temporal Transformer over 24 h → second graph-attention step
→ direct multi-horizon head). *Robust* variants are trained with random station dropout so they keep working when
sensors fail. The **combined forecast** averages the GAT-Transformer, Robust GAT-Transformer and Transformer.

**Graph.** Each station connects to its {C.KNN} nearest stations; edges are weighted by a Gaussian kernel of
Haversine distance, used as an attention prior the model can override.

**Data & evaluation.** Hourly PM2.5 from the XKDR Air Quality API (2023 – Aug 2025) and CPCB hourly files
(2025 – Sep 2026). Hold-out models are trained on Jan 2023 – Jun 2024, validated on Jul – Sep 2024 and tested on
Oct 2024 – Sep 2026. The live models are then trained on all data up to Sep 2026. Scalers are fitted on training data
only; missing inputs are filled causally; models are trained and scored only on real observed values.

**Live readings.** WAQI provides PM2.5 as a US-EPA AQI value, converted back to µg/m³ (capped near 500).
Colour categories use the Indian national PM2.5 breakpoints (Good ≤30, Satisfactory ≤60, Moderate ≤90, Poor ≤120,
Very Poor ≤250, Severe >250 µg/m³).

**Run it yourself.** `python -m streamlit run app/streamlit_app.py`; keep `python src/live.py --loop` running to
save readings and score forecasts continuously.
""")
