"""
events.py
A lightweight event layer. An Event has a location, a time window, and a crowd
magnitude. Its congestion impact on a road link decays with distance from the
venue and ramps up as the event start/end time approaches (people arriving and
leaving). Used both to inject events into the generated history and to drive
"what-if" scenarios in the simulator.
"""

import sys
import pathlib
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

import config as C


@dataclass
class Event:
    name: str
    lat: float
    lon: float
    crowd: float          # 0-1, how big
    radius_m: float       # spatial reach
    start_step: int       # global timestep index when crowd peaks (arrival)
    duration_steps: int   # how long the surge lasts


def _haversine_m(lat1, lon1, lat2, lon2):
    r = 6_371_000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def event_intensity(events, links, timestamps_idx):
    """
    Build a [T, N] array of event-driven extra congestion in [0, 1].

    events           : list[Event]
    links            : DataFrame with 'lat','lon'
    timestamps_idx   : 1-D array of global timestep indices (length T)
    """
    T = len(timestamps_idx)
    N = len(links)
    out = np.zeros((T, N), dtype=np.float32)
    if not events:
        return out

    lat = links["lat"].values
    lon = links["lon"].values
    for ev in events:
        d = _haversine_m(lat, lon, ev.lat, ev.lon)           # [N]
        spatial = ev.crowd * np.exp(-(d / ev.radius_m) ** 2)  # [N]
        # temporal bump: triangular ramp centred on the event window
        half = max(ev.duration_steps / 2.0, 1.0)
        rel = (timestamps_idx - ev.start_step) / half          # [T]
        temporal = np.clip(1.0 - np.abs(rel), 0.0, 1.0)        # [T]
        out += np.outer(temporal, spatial).astype(np.float32)

    return np.clip(out, 0.0, 1.0)


def make_event(keyword, start_step, duration_steps=8):
    """Create an Event from a venue keyword (see config.EVENT_VENUES)."""
    key = keyword.lower().strip()
    if key not in C.EVENT_VENUES:
        return None
    name, lat, lon, crowd, radius = C.EVENT_VENUES[key]
    return Event(name, lat, lon, crowd, radius, start_step, duration_steps)
