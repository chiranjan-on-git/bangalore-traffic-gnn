"""
traffic_generator.py
Generates realistic-but-synthetic traffic on the real Bangalore road graph.

Why synthetic: Bangalore has no free public real-time traffic-sensor feed, so we
model congestion from first principles using patterns the city actually exhibits:
  * twin weekday peaks (~9:00 and ~18:30), milder weekend profile
  * arterials (ORR/trunk roads) carry more load than minor roads
  * named hotspots (Silk Board, Marathahalli, KR Puram ...) run hotter
  * rain slows everything down
  * events create localised surges
  * SPATIAL PROPAGATION - congestion diffuses to connected segments, so a
    graph model can exploit neighbour structure (a non-graph model cannot).

Everything except the traffic itself is real (roads from OSM, weather live).
The feature-assembly here is reused by the simulator so train/inference match.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from scipy import sparse

import config as C

FEATURE_NAMES = [
    "speed_ratio", "tod_sin", "tod_cos", "weekend", "weather",
    "event", "speed_kmh", "length", "class_ord", "hotspot",
]


# --------------------------------------------------------------------------- #
# Time-of-day demand profile
# --------------------------------------------------------------------------- #
def _gauss(h, c, w):
    return np.exp(-(((h - c) / w) ** 2))


def demand_profile(timestamps_idx):
    """Return (demand[T], tod_sin[T], tod_cos[T], weekend[T])."""
    step_in_day = timestamps_idx % C.STEPS_PER_DAY
    hour = step_in_day * C.TIME_STEP_MIN / 60.0
    dow = (timestamps_idx // C.STEPS_PER_DAY) % 7  # START_DATE is a Monday
    weekend = (dow >= 5).astype(np.float32)

    daytime = ((hour >= 7) & (hour <= 22)).astype(np.float32)
    weekday_demand = (0.15 * daytime
                      + 0.55 * _gauss(hour, 9.0, 1.6)
                      + 0.70 * _gauss(hour, 18.5, 1.8))
    weekend_demand = (0.10 * daytime
                      + 0.25 * _gauss(hour, 12.0, 3.0)
                      + 0.30 * _gauss(hour, 19.0, 2.5))
    demand = np.where(weekend > 0, weekend_demand, weekday_demand).astype(np.float32)

    ang = 2 * np.pi * step_in_day / C.STEPS_PER_DAY
    return demand, np.sin(ang).astype(np.float32), np.cos(ang).astype(np.float32), weekend


# --------------------------------------------------------------------------- #
# Spatial diffusion operator (row-normalised neighbour averaging)
# --------------------------------------------------------------------------- #
def diffusion_matrix(edge_index, n):
    if edge_index.shape[1] == 0:
        return sparse.identity(n, format="csr", dtype=np.float32)
    src, dst = edge_index[0], edge_index[1]
    data = np.ones(len(src), dtype=np.float32)
    A = sparse.coo_matrix((data, (dst, src)), shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    Dinv = sparse.diags((1.0 / deg).astype(np.float32))
    return (Dinv @ A).tocsr()  # row-normalised: M @ x = neighbour mean


# --------------------------------------------------------------------------- #
# Core congestion model
# --------------------------------------------------------------------------- #
def compute_speed_ratio(links, edge_index, timestamps_idx, weather_arr,
                        event_int, rng, with_noise=True, diffusion_rounds=2):
    T = len(timestamps_idx)
    n = len(links)

    demand, _, _, _ = demand_profile(timestamps_idx)              # [T]
    class_sens = 0.5 + 0.125 * links["class_ord"].values          # [N] in 0.5..1.0
    hotspot = links["hotspot"].values.astype(np.float32)          # [N]
    weather_pen = np.array([C.WEATHER_PENALTY[int(w)] for w in weather_arr],
                           dtype=np.float32)                      # [T]

    # base congestion: demand x road sensitivity (outer product)
    cong = np.outer(demand, class_sens).astype(np.float32)        # [T, N]
    # hotspots scale with demand
    cong += np.outer(demand, hotspot) * 0.9
    # weather hits everyone
    cong += weather_pen[:, None]
    # events (already [T, N])
    cong += 1.1 * event_int

    # spatial propagation
    M = diffusion_matrix(edge_index, n)
    for _ in range(diffusion_rounds):
        neigh = (M @ cong.T).T                                    # [T, N]
        cong = 0.7 * cong + 0.3 * neigh

    # temporal smoothing (AR1) so consecutive steps are correlated
    rho = 0.6
    for t in range(1, T):
        cong[t] = rho * cong[t - 1] + (1 - rho) * cong[t]

    if with_noise:
        cong += rng.normal(0.0, 0.04, size=cong.shape).astype(np.float32)

    speed_ratio = np.clip(1.0 - cong, 0.10, 1.0).astype(np.float32)
    return speed_ratio


# --------------------------------------------------------------------------- #
# Feature assembly (shared by training data + simulator)
# --------------------------------------------------------------------------- #
def assemble_features(links, timestamps_idx, weather_arr, event_int, speed_ratio):
    T = len(timestamps_idx)
    n = len(links)
    _, tod_sin, tod_cos, weekend = demand_profile(timestamps_idx)

    static = np.zeros((n, 4), dtype=np.float32)
    static[:, 0] = links["speed_kmh"].values / 80.0
    static[:, 1] = np.clip(links["length_m"].values / 500.0, 0, 2)
    static[:, 2] = links["class_ord"].values / 4.0
    static[:, 3] = links["hotspot"].values

    X = np.zeros((T, n, len(FEATURE_NAMES)), dtype=np.float32)
    X[:, :, 0] = speed_ratio
    X[:, :, 1] = tod_sin[:, None]
    X[:, :, 2] = tod_cos[:, None]
    X[:, :, 3] = weekend[:, None]
    X[:, :, 4] = (weather_arr / 3.0)[:, None]
    X[:, :, 5] = event_int
    X[:, :, 6:10] = static[None, :, :]
    return X


# --------------------------------------------------------------------------- #
# Synthetic weather sampler for the generated history (live data uses weather.py)
# --------------------------------------------------------------------------- #
def sample_weather(timestamps_idx, rng):
    """Bangalore-ish: occasional rain spells, more likely in the evening."""
    T = len(timestamps_idx)
    weather = np.zeros(T, dtype=np.float32)
    t = 0
    while t < T:
        if rng.random() < 0.04:  # start of a rain spell
            spell = rng.integers(2, 10)
            intensity = rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
            weather[t:t + spell] = intensity
            t += spell
        else:
            t += 1
    return weather
