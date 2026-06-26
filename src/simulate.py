"""
simulate.py
The scenario engine + natural-language query parser that powers the interactive
"what-if" output. It builds a model-ready input window from a scenario, runs the
GNN, and returns predicted speed ratios per road segment for the forecast hour.

Example query:
  "If a concert happens at Chinnaswamy and it rains during evening rush,
   which roads get congested?"
"""

import re
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch

import config as C
from src import events as ev
from src import traffic_generator as gen


# --------------------------------------------------------------------------- #
# Scenario -> model input -> prediction
# --------------------------------------------------------------------------- #
def _step_for_hour(hour, weekend=False):
    """Global timestep index for a given clock hour on a representative day."""
    base_day = 5 if weekend else 2  # a Sat or a Wed inside the generated range
    return base_day * C.STEPS_PER_DAY + int(round(hour * 60 / C.TIME_STEP_MIN))


def build_scenario_window(links, edge_index, hour, weekend, weather_intensity,
                          event_keywords):
    """
    Construct the IN_STEPS input window leading up to `hour`, with the chosen
    conditions held constant across the window. Returns (X_window, target_step).
    """
    target_step = _step_for_hour(hour, weekend)
    start = target_step - C.IN_STEPS
    idx = np.arange(start, target_step)                       # [IN_STEPS]

    weather_arr = np.full(C.IN_STEPS, weather_intensity, dtype=np.float32)

    events = []
    for kw in event_keywords:
        e = ev.make_event(kw, start_step=target_step - C.IN_STEPS // 2,
                          duration_steps=C.IN_STEPS)
        if e:
            events.append(e)
    event_int = ev.event_intensity(events, links, idx)        # [IN_STEPS, N]

    rng = np.random.default_rng(0)
    speed_ratio = gen.compute_speed_ratio(
        links, edge_index, idx, weather_arr, event_int, rng, with_noise=False
    )
    X = gen.assemble_features(links, idx, weather_arr, event_int, speed_ratio)
    return X, events


def predict(model, ckpt, links, edge_index_t, X_window):
    """Run the GNN. Returns predicted speed ratio per link for the final horizon."""
    full = predict_full(model, ckpt, edge_index_t, X_window)   # [OUT, N]
    return full.mean(axis=0)                                    # average over horizon


def predict_full(model, ckpt, edge_index_t, X_window):
    """Return the full [OUT_STEPS, N] forecast."""
    mean = ckpt["mean"]
    std = ckpt["std"]
    x = (X_window - mean) / std
    x = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)    # [1, IN, N, F]
    with torch.no_grad():
        pred = model(x, edge_index_t)                          # [1, OUT, N]
    return pred.squeeze(0).numpy()                             # [OUT, N]


def run_scenario(model, ckpt, links, edge_index_t, edge_index_np,
                 hour=18.5, weekend=False, weather_intensity=0, event_keywords=None):
    event_keywords = event_keywords or []
    X_window, events = build_scenario_window(
        links, edge_index_np, hour, weekend, weather_intensity, event_keywords
    )
    speed_pred = predict(model, ckpt, links, edge_index_t, X_window)
    return speed_pred, events


# --------------------------------------------------------------------------- #
# Natural-language query parser (rule-based, lightweight)
# --------------------------------------------------------------------------- #
def parse_query(text):
    """Map a free-text scenario to {hour, weekend, weather_intensity, events}."""
    t = text.lower()

    # weather
    weather = 0
    heavy = re.search(r"heavy rain|downpour|storm|thunder|torrential", t) or \
        ("rain" in t and re.search(r"heav|intense|pour", t))
    if heavy:
        weather = 3
    elif re.search(r"\brain|wet|monsoon\b", t):
        weather = 2
    elif re.search(r"drizzle|light rain|shower", t):
        weather = 1
    elif re.search(r"clear|sunny|dry|no rain", t):
        weather = 0

    # day type
    weekend = bool(re.search(r"weekend|saturday|sunday", t))

    # time of day
    hour = 18.5  # default to evening rush
    m = re.search(r"(\d{1,2})\s*(am|pm)", t)
    if m:
        h = int(m.group(1)) % 12
        hour = h + (12 if m.group(2) == "pm" else 0)
    elif re.search(r"morning|9 ?am|office", t):
        hour = 9.0
    elif re.search(r"afternoon|noon|lunch", t):
        hour = 13.0
    elif re.search(r"night|late", t):
        hour = 22.0
    elif re.search(r"evening|rush|peak", t):
        hour = 18.5

    # events
    found = []
    for kw in C.EVENT_VENUES:
        if kw in t and C.EVENT_VENUES[kw][0] not in {
            C.EVENT_VENUES[k][0] for k in found
        }:
            found.append(kw)

    return {
        "hour": hour,
        "weekend": weekend,
        "weather_intensity": weather,
        "event_keywords": found,
    }


def describe(scenario):
    parts = [f"{int(scenario['hour']):02d}:00",
             "weekend" if scenario["weekend"] else "weekday",
             C.WEATHER_LABELS[scenario["weather_intensity"]]]
    if scenario["event_keywords"]:
        venues = sorted({C.EVENT_VENUES[k][0] for k in scenario["event_keywords"]})
        parts.append("event: " + ", ".join(venues))
    return " | ".join(parts)
