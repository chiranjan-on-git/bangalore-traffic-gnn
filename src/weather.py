"""
weather.py
Fetches real current weather for Bangalore from the Open-Meteo API
(free, no API key required). Used by the Streamlit app for the "now" scenario.

Training data uses synthetically sampled weather (see traffic_generator.py),
so this module is only needed live; it fails soft to "Clear" if offline.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import requests

import config as C

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def _precip_to_intensity(precip_mm):
    if precip_mm is None or precip_mm <= 0.05:
        return 0
    if precip_mm < 0.5:
        return 1
    if precip_mm < 4.0:
        return 2
    return 3


def fetch_current_weather(timeout=8):
    """Return a dict: {intensity 0-3, label, precip_mm, temp_c, source}."""
    lat, lon = C.NETWORK_CENTER
    try:
        resp = requests.get(
            OPEN_METEO_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,precipitation",
                "timezone": "Asia/Kolkata",
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        cur = resp.json().get("current", {})
        precip = cur.get("precipitation")
        temp = cur.get("temperature_2m")
        intensity = _precip_to_intensity(precip)
        return {
            "intensity": intensity,
            "label": C.WEATHER_LABELS[intensity],
            "precip_mm": precip,
            "temp_c": temp,
            "source": "Open-Meteo (live)",
        }
    except Exception as exc:  # noqa: BLE001 - fail soft when offline
        return {
            "intensity": 0,
            "label": C.WEATHER_LABELS[0],
            "precip_mm": 0.0,
            "temp_c": None,
            "source": f"fallback (offline: {type(exc).__name__})",
        }


if __name__ == "__main__":
    print(fetch_current_weather())
