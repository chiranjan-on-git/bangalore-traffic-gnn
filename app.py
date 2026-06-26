"""
app.py  -  Interactive Simulation Output

    streamlit run app.py

A Bangalore congestion simulator. Pick conditions (or type a natural-language
"what-if" query), and the trained GNN predicts speed across every arterial road
segment for the next hour, drawn on the map (green = free flowing, red = jammed).
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
import streamlit as st
import folium
from streamlit_folium import st_folium

import config as C
from src import road_network
from src.model import load_checkpoint
from src import simulate as sim
from src import weather as wx

st.set_page_config(page_title="Bangalore Traffic Flow Predictor",
                   layout="wide", page_icon="🚦")


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_everything():
    links, edge_index_np = road_network.load()
    model, ckpt = load_checkpoint("cpu")
    edge_index_t = torch.from_numpy(edge_index_np).long()
    return links, edge_index_np, edge_index_t, model, ckpt


def ratio_to_color(r):
    """Speed ratio 1.0 -> green, 0.5 -> orange, 0.1 -> red."""
    r = float(np.clip(r, 0.1, 1.0))
    if r > 0.6:
        f = (r - 0.6) / 0.4          # 0..1 across orange->green
        red = int(255 * (1 - f)); grn = 200
    else:
        f = (r - 0.1) / 0.5          # 0..1 across red->orange
        red = 230; grn = int(180 * f)
    return f"#{red:02x}{grn:02x}30"


def build_map(links, speed_pred, events):
    lat, lon = C.NETWORK_CENTER
    m = folium.Map(location=[lat, lon], zoom_start=12, tiles="cartodbpositron")
    for i, row in links.iterrows():
        geom = row["geometry"]
        if geom is None or geom.is_empty:
            continue
        coords = [(float(y), float(x)) for x, y in geom.coords]  # (lat, lon)
        r = float(speed_pred[i])
        folium.PolyLine(
            coords, color=ratio_to_color(r),
            weight=float(2 + 4 * (1 - r)), opacity=0.85,
        ).add_to(m)

    for ev_obj in events:
        folium.CircleMarker(
            [float(ev_obj.lat), float(ev_obj.lon)], radius=8, color="#1f4ed8",
            fill=True, fill_opacity=0.9,
            popup=f"Event: {ev_obj.name}",
        ).add_to(m)
    return m


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
st.title("🚦 Bangalore Traffic Flow Prediction System")
st.caption("Spatio-temporal GNN forecasting congestion across the Central / East "
           "Bangalore arterial network (ORR, Silk Board, Marathahalli, KR Puram).")

try:
    links, edge_index_np, edge_index_t, model, ckpt = load_everything()
except FileNotFoundError:
    st.error("Model or data not found. Run these first:\n\n"
             "```\npython scripts/01_build_network.py\n"
             "python scripts/02_generate_data.py\n"
             "python scripts/03_train.py\n```")
    st.stop()

with st.sidebar:
    st.header("Scenario")
    mode = st.radio("Input mode", ["Manual controls", "Natural-language query"])

    if mode == "Manual controls":
        day_type = st.selectbox("Day", ["Weekday", "Weekend"])
        hour = st.slider("Time of day (hour)", 0.0, 23.5, 18.5, 0.5)
        weather_label = st.selectbox(
            "Weather", [C.WEATHER_LABELS[i] for i in range(4)], index=0)
        weather_intensity = [k for k, v in C.WEATHER_LABELS.items()
                             if v == weather_label][0]
        venue_names = sorted({v[0] for v in C.EVENT_VENUES.values()})
        chosen_venues = st.multiselect("Active events", venue_names)
        # map display names back to keywords
        event_keywords = [k for k, v in C.EVENT_VENUES.items()
                          if v[0] in chosen_venues]
        event_keywords = list(dict.fromkeys(event_keywords))
        scenario = {
            "hour": hour, "weekend": day_type == "Weekend",
            "weather_intensity": weather_intensity,
            "event_keywords": event_keywords,
        }
        if st.button("Use live Bangalore weather"):
            live = wx.fetch_current_weather()
            st.info(f"Live: {live['label']} ({live['source']})")
            scenario["weather_intensity"] = live["intensity"]
    else:
        st.write("Try: *If a concert happens at Chinnaswamy and it rains "
                 "during evening rush, which roads get congested?*")
        query = st.text_area("Your what-if question",
                             "If there's a cricket match at Chinnaswamy and it "
                             "rains during evening rush, which roads get congested?")
        scenario = sim.parse_query(query)

    show_delta = st.checkbox("Show change vs a clear, event-free baseline", False)

st.markdown(f"**Scenario:** {sim.describe(scenario)}")

# Run the model.
speed_pred, events = sim.run_scenario(
    model, ckpt, links, edge_index_t, edge_index_np,
    hour=scenario["hour"], weekend=scenario["weekend"],
    weather_intensity=scenario["weather_intensity"],
    event_keywords=scenario["event_keywords"],
)

display_vals = speed_pred
if show_delta:
    base_pred, _ = sim.run_scenario(
        model, ckpt, links, edge_index_t, edge_index_np,
        hour=scenario["hour"], weekend=scenario["weekend"],
        weather_intensity=0, event_keywords=[],
    )
    # negative = scenario is slower than baseline (worse)
    display_vals = 1.0 + (speed_pred - base_pred)  # recolor: <1 worse, >1 better

col_map, col_side = st.columns([3, 1])

with col_map:
    fmap = build_map(links, display_vals, events)
    st_folium(fmap, width=None, height=560, returned_objects=[])

with col_side:
    avg = float(speed_pred.mean())
    jammed = int((speed_pred < 0.4).sum())
    st.metric("Network avg speed ratio", f"{avg:.2f}",
              help="1.0 = free flow, 0.1 = gridlock")
    st.metric("Segments below 40% speed", f"{jammed} / {len(links)}")

    worst = (pd.DataFrame({
        "road": links.get("name", pd.Series(["segment"] * len(links))).astype(str),
        "lat": links["lat"].round(4),
        "lon": links["lon"].round(4),
        "speed_ratio": speed_pred.round(2),
    }).sort_values("speed_ratio").head(10).reset_index(drop=True))
    st.subheader("Worst-hit segments")
    st.dataframe(worst[["lat", "lon", "speed_ratio"]], hide_index=True,
                 use_container_width=True)

# Forecast curve for the single most-congested segment.
worst_link = int(np.argmin(speed_pred))
X_window, _ = sim.build_scenario_window(
    links, edge_index_np, scenario["hour"], scenario["weekend"],
    scenario["weather_intensity"], scenario["event_keywords"])
horizon = sim.predict_full(model, ckpt, edge_index_t, X_window)  # [OUT, N]
mins = [(k + 1) * C.TIME_STEP_MIN for k in range(C.OUT_STEPS)]
curve = pd.DataFrame({"minutes ahead": mins,
                      "predicted speed ratio": horizon[:, worst_link].round(3)})
st.subheader(f"1-hour forecast for the most congested segment "
             f"(near {links.iloc[worst_link]['lat']:.3f}, "
             f"{links.iloc[worst_link]['lon']:.3f})")
st.line_chart(curve, x="minutes ahead", y="predicted speed ratio")

st.caption("Roads are coloured by predicted speed: green = free flowing, "
           "red = congested. Traffic values are model-generated on the real OSM "
           "network; weather is live from Open-Meteo.")