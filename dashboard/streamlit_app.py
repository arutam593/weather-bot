"""Streamlit dashboard with live Open-Meteo forecasts + Polymarket tab."""
from __future__ import annotations

import asyncio
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit_authenticator as stauth
import yaml

from src.ingestion.geographic import GeographicAdapter
from src.models.short_term import ShortTermModel
from src.models.mos import MOSCorrector
from src.models.ensemble import Ensemble
from src.processing.features import FeatureBuilder, TARGET_VARIABLES
from src.processing.nlp import WeatherSignalExtractor
from src.ingestion.base import SourceReading, utc_now
from src.polymarket import analyze as polymarket_analyze

st.set_page_config(page_title="Weather Bot", layout="wide", page_icon="⛅")

# ─── auth gate ───────────────────────────────────────────
# Read auth config from Streamlit Cloud secrets if available, else local file
if "auth" in st.secrets and "config_yaml" in st.secrets["auth"]:
    auth_config = yaml.safe_load(st.secrets["auth"]["config_yaml"])
else:
    with open("config/auth.yaml") as f:
        auth_config = yaml.safe_load(f)

authenticator = stauth.Authenticate(
    auth_config["credentials"],
    auth_config["cookie"]["name"],
    auth_config["cookie"]["key"],
    auth_config["cookie"]["expiry_days"],
)

authenticator.login(location="main")

if st.session_state.get("authentication_status") is False:
    st.error("Wrong username or password.")
    st.stop()
elif st.session_state.get("authentication_status") is None:
    st.warning("Please log in to continue.")
    st.stop()

# Logged in beyond this point
with st.sidebar:
    st.success(f"Logged in as **{st.session_state['name']}**")
    authenticator.logout(location="sidebar")

st.markdown("""
<style>
.metric-card {background:#fafafa;padding:.8rem;border-radius:6px;border:1px solid #eee;text-align:center}
.pill-ok {background:#d4edda;color:#155724;padding:.25rem .6rem;border-radius:12px;font-size:.85rem}
.pill-bad {background:#f8d7da;color:#721c24;padding:.25rem .6rem;border-radius:12px;font-size:.85rem}
.section-h {color:#444;border-bottom:2px solid #eee;padding-bottom:.3rem;margin-top:1.5rem}
.live-dot {display:inline-block;width:10px;height:10px;background:#2ca02c;border-radius:50%;margin-right:6px;animation:pulse 1.4s infinite}
@keyframes pulse {0%{opacity:1}50%{opacity:.3}100%{opacity:1}}
</style>
""", unsafe_allow_html=True)


HOURLY_VARS = ("temperature_2m,relative_humidity_2m,precipitation,"
               "wind_speed_10m,wind_direction_10m,pressure_msl,cloud_cover")


@st.cache_data(ttl=3600)
def geocode_city(query):
    if not query or len(query.strip()) < 2:
        return []
    try:
        r = httpx.get("https://geocoding-api.open-meteo.com/v1/search",
                      params={"name": query, "count": 5, "language": "en"},
                      timeout=10.0)
        r.raise_for_status()
        out = []
        for c in r.json().get("results", []) or []:
            parts = [c["name"]]
            if c.get("admin1"):
                parts.append(c["admin1"])
            if c.get("country"):
                parts.append(c["country"])
            out.append({"label": ", ".join(parts), "lat": c["latitude"],
                        "lon": c["longitude"], "name": c["name"]})
        return out
    except Exception as e:
        st.error("Geocoding failed: " + str(e))
        return []


def vars_from_om(block, i):
    return {
        "temp_c":       block["temperature_2m"][i],
        "humidity_pct": block["relative_humidity_2m"][i],
        "precip_mm":    block["precipitation"][i],
        "wind_ms":      block["wind_speed_10m"][i] / 3.6,
        "wind_dir_deg": block["wind_direction_10m"][i],
        "pressure_hpa": block["pressure_msl"][i],
        "cloud_pct":    block["cloud_cover"][i],
    }


@st.cache_data(ttl=600)
def fetch_open_meteo(lat, lon):
    from datetime import datetime, timedelta, timezone
    end = datetime.now(timezone.utc).date() - timedelta(days=2)
    start = end - timedelta(days=30)
    hist = httpx.get("https://archive-api.open-meteo.com/v1/archive",
                     params={"latitude": lat, "longitude": lon,
                             "start_date": start.isoformat(),
                             "end_date": end.isoformat(),
                             "hourly": HOURLY_VARS, "timezone": "UTC"},
                     timeout=30.0).json()
    fc = httpx.get("https://api.open-meteo.com/v1/forecast",
                   params={"latitude": lat, "longitude": lon,
                           "hourly": HOURLY_VARS, "forecast_days": 7,
                           "timezone": "UTC", "models": "best_match"},
                   timeout=30.0).json()
    h = hist["hourly"]
    hist_df = pd.DataFrame({
        "valid_time":   pd.to_datetime(h["time"], utc=True),
        "temp_c":       h["temperature_2m"],
        "humidity_pct": h["relative_humidity_2m"],
        "precip_mm":    h["precipitation"],
        "wind_ms":      [v / 3.6 for v in h["wind_speed_10m"]],
        "pressure_hpa": h["pressure_msl"],
        "cloud_pct":    h["cloud_cover"],
    }).dropna()
    return hist_df, fc


def df_to_readings(df, lat, lon, source):
    out = []
    var_cols = [c for c in df.columns if c != "valid_time"]
    for row in df.itertuples():
        out.append(SourceReading(
            source=source, fetched_at=utc_now(),
            valid_time=getattr(row, "valid_time"),
            lead_hours=0.0, lat=lat, lon=lon,
            variables={c: float(getattr(row, c))
                       for c in var_cols if pd.notna(getattr(row, c))},
            reliability_prior=1.0,
        ))
    return out


async def run_forecast(lat, lon, name):
    geo = await GeographicAdapter().fetch(lat, lon)
    hist_df, fc_data = fetch_open_meteo(lat, lon)
    now = utc_now()

    fc_readings = []
    fc_rows = []
    h = fc_data["hourly"]
    times = pd.to_datetime(h["time"], utc=True)
    for i, t in enumerate(times):
        if t < now - pd.Timedelta(hours=1):
            continue
        v = vars_from_om(h, i)
        fc_readings.append(SourceReading(
            source="open_meteo", fetched_at=now, valid_time=t,
            lead_hours=max(0.0, (t - now).total_seconds() / 3600),
            lat=lat, lon=lon, variables=v, reliability_prior=1.0))
        fc_rows.append({"valid_time": t, **v})

    builder = FeatureBuilder(geo=geo, normals={})
    hist_df = hist_df.set_index("valid_time")
    train_readings = df_to_readings(hist_df.reset_index(), lat, lon,
                                     source="open_meteo")
    truth = hist_df[["temp_c", "precip_mm", "wind_ms"]]
    nlp_baseline = WeatherSignalExtractor({}).extract([], target_location=name)
    train_frame = builder.build_training(train_readings, truth, nlp_baseline)
    train_frame.X = train_frame.X.dropna()
    train_frame.y = train_frame.y.loc[train_frame.X.index]

    short = ShortTermModel(horizon_hours=168).fit(train_frame.X, train_frame.y)
    mos = MOSCorrector(alpha=1.0)
    mos.fit(train_frame.X, train_frame.y, location_key=str(lat) + "," + str(lon))

    inf_readings = (df_to_readings(hist_df.tail(48).reset_index(),
                                     lat, lon, source="open_meteo")
                    + fc_readings)
    inf_frame = builder.build_inference(inf_readings, nlp_baseline)
    inf_frame.X = inf_frame.X.ffill().fillna(0)
    sim_now = pd.Timestamp(now)
    future_X = inf_frame.X[inf_frame.X.index > sim_now]

    short_preds = short.predict(future_X, now=sim_now)
    consensus = pd.DataFrame({
        v: future_X["consensus__" + v]
        for v in TARGET_VARIABLES
        if "consensus__" + v in future_X.columns
    })
    mos_consensus = pd.DataFrame({
        v: mos.correct(future_X, v, str(lat) + "," + str(lon))
        for v in TARGET_VARIABLES
        if (v, str(lat) + "," + str(lon)) in mos.models
    })
    climate_std = {v: float(train_frame.y[v].std())
                   for v in train_frame.y.columns}
    ensemble = Ensemble(climate_std=climate_std)
    ens_preds = ensemble.combine(short_preds, [], consensus,
                                  mos_consensus=mos_consensus, now=sim_now)
    return geo, ens_preds


# ─── sidebar ─────────────────────────────────────────────

with st.sidebar:
    st.title("⛅ Weather Bot")
    st.caption("7-day forecasts + Polymarket comparison")

    st.markdown("### 🔎 Search any city")
    query = st.text_input("Type and press Enter", value="",
                          placeholder="Madrid, Tokyo, Lagos…")
    selected = None
    if query:
        candidates = geocode_city(query)
        if candidates:
            choice = st.radio("Choose a match",
                              [c["label"] for c in candidates], index=0,
                              label_visibility="collapsed")
            selected = next(c for c in candidates if c["label"] == choice)
        else:
            st.warning("No matches.")

    if not selected:
        lat = st.number_input("Latitude", value=40.4168, format="%.4f")
        lon = st.number_input("Longitude", value=-3.7038, format="%.4f")
        loc_name = st.text_input("Name", value="Custom")
    else:
        lat, lon, loc_name = selected["lat"], selected["lon"], selected["name"]
        st.caption("📍 " + selected["label"])

    st.divider()
    live_on = st.toggle("⚡ Auto-refresh", value=False)
    refresh_seconds = st.slider("Interval (s)", 60, 600, 180, 30,
                                 disabled=not live_on)
    show_map = st.checkbox("Show map", value=True)
    run_clicked = st.button("🌤️ Run prediction now", type="primary",
                             use_container_width=True)


# ─── tabs ────────────────────────────────────────────────

tab_forecast, tab_polymarket = st.tabs(["🌤️ Forecast", "📊 Polymarket"])


# ─── Polymarket tab ──────────────────────────────────────

with tab_polymarket:
    st.markdown("### Weather markets on Polymarket")
    st.caption("Statistics only. Compares market-implied probability "
               "against the bot's forecast. No bet recommendations.")
    pm_run = st.button("Refresh Polymarket data", key="pm_refresh")
    if pm_run:
        with st.spinner("Fetching markets and running forecasts… "
                        "(takes 1-3 minutes)"):
            try:
                pm_df = asyncio.run(polymarket_analyze())
            except Exception as e:
                st.error("Polymarket fetch failed: " + str(e))
                pm_df = None
        if pm_df is None or len(pm_df) == 0:
            st.info("No active weather markets found right now.")
        else:
            st.dataframe(pm_df, use_container_width=True, hide_index=True)
            st.caption("**gap** = bot's probability minus market's implied "
                       "probability. Positive means the bot thinks YES is "
                       "underpriced; negative means overpriced.")
    else:
        st.info("Click **Refresh Polymarket data** to fetch the table. "
                "Each refresh takes 1–3 minutes.")


# ─── Forecast tab ────────────────────────────────────────

with tab_forecast:
    hcol, scol = st.columns([3, 1])
    with hcol:
        st.title("Forecast for **" + loc_name + "**")
    with scol:
        if live_on:
            st.markdown(
                "<div style='text-align:right;padding-top:1.4rem'>"
                "<span class='live-dot'></span><b>LIVE</b> · " +
                str(refresh_seconds) + "s</div>", unsafe_allow_html=True)

    if not run_clicked and not live_on:
        st.info("👈 Search a city, then **Run prediction**.")
        if show_map:
            st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), zoom=4)
    else:
        with st.spinner("Fetching live data and running 7-day forecast for "
                        + loc_name + "…"):
            try:
                geo, preds = asyncio.run(run_forecast(lat, lon, loc_name))
            except Exception as e:
                st.error("Forecast failed: " + str(e))
                st.stop()

        st.caption("Last updated: "
                   + pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                   + " UTC")

        cl, cr = st.columns([2, 1])
        with cl:
            if show_map:
                st.markdown("<div class='section-h'>📍 Location</div>",
                            unsafe_allow_html=True)
                st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), zoom=8)
        with cr:
            st.markdown("<div class='section-h'>🌍 Geography</div>",
                        unsafe_allow_html=True)
            g = geo.as_dict()
            c1, c2 = st.columns(2)
            c1.metric("Elevation", str(int(g["elevation_m"])) + " m")
            c2.metric("Coast", str(int(g["coast_distance_km"])) + " km")
            c3, c4 = st.columns(2)
            c3.metric("Urban heat", "{:.2f}".format(g["urban_heat_index"]))
            c4.metric("Ruggedness",
                       str(int(g["terrain_ruggedness_m"])) + " m")

        st.markdown("<div class='section-h'>📈 7-day forecast</div>",
                    unsafe_allow_html=True)

        by_var = defaultdict(list)
        for p in preds:
            by_var[p.variable].append(p)

        unit_map  = {"temp_c": "°C", "precip_mm": "mm", "wind_ms": "m/s"}
        nice_map  = {"temp_c": "Temperature", "precip_mm": "Precipitation",
                     "wind_ms": "Wind speed"}
        color_map = {"temp_c": "#e74c3c", "precip_mm": "#3498db",
                     "wind_ms": "#27ae60"}
        rgb_map   = {"temp_c": "231,76,60", "precip_mm": "52,152,219",
                     "wind_ms": "39,174,96"}

        cols = st.columns(len(by_var) or 1)
        for col, (var, ps) in zip(cols, by_var.items()):
            ps = sorted(ps, key=lambda p: p.valid_time)
            df = pd.DataFrame({
                "t":     [p.valid_time for p in ps],
                "point": [p.point for p in ps],
                "lower": [p.lower for p in ps],
                "upper": [p.upper for p in ps],
                "conf":  [p.confidence_pct for p in ps],
            })
            color = color_map.get(var, "#7f7f7f")
            rgb   = rgb_map.get(var, "127,127,127")
            unit  = unit_map.get(var, "")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(df["t"]) + list(df["t"][::-1]),
                y=list(df["upper"]) + list(df["lower"][::-1]),
                fill="toself", fillcolor="rgba(" + rgb + ",0.2)",
                line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
                showlegend=False))
            fig.add_trace(go.Scatter(
                x=df["t"], y=df["point"], mode="lines+markers",
                line=dict(color=color, width=2.5),
                marker=dict(size=6, color=color,
                            line=dict(color="white", width=1)),
                customdata=list(zip(df["lower"], df["upper"], df["conf"])),
                hovertemplate=(
                    "<b>%{x|%a %b %d, %H:%M}</b><br>"
                    "Predicted: <b>%{y:.1f} " + unit + "</b><br>"
                    "Range: %{customdata[0]:.1f} – %{customdata[1]:.1f} " + unit + "<br>"
                    "Confidence: %{customdata[2]:.0f}%"
                    "<extra></extra>"),
                showlegend=False))
            fig.update_layout(
                title=nice_map.get(var, var) + " (" + unit + ")",
                height=340, margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title="", yaxis_title=unit,
                showlegend=False, plot_bgcolor="white",
                hovermode="x unified",
                xaxis=dict(showgrid=True, gridcolor="#f0f0f0",
                           showspikes=True, spikemode="across",
                           spikesnap="cursor", spikecolor="#999",
                           spikethickness=1, spikedash="dot"),
                yaxis=dict(showgrid=True, gridcolor="#f0f0f0"))
            col.plotly_chart(fig, use_container_width=True)


if live_on and run_clicked:
    time.sleep(refresh_seconds)
    st.rerun()