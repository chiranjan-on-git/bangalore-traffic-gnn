"""
Step 2 - Generate the realistic synthetic traffic history on the road graph.
No internet needed.

    python scripts/02_generate_data.py
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

import config as C
from src import road_network
from src import events as ev
from src import traffic_generator as gen


def sample_history_events(rng, total_steps):
    """Sprinkle a handful of events across the history (evenings, mostly)."""
    venues = list({v[0]: k for k, v in C.EVENT_VENUES.items()}.values())
    events = []
    n_events = max(4, C.N_DAYS // 3)
    for _ in range(n_events):
        day = rng.integers(0, C.N_DAYS)
        # evening start ~ 18:00-20:00
        hour = rng.uniform(18.0, 20.0)
        start = day * C.STEPS_PER_DAY + int(hour * 60 / C.TIME_STEP_MIN)
        if start >= total_steps:
            continue
        kw = rng.choice(venues)
        e = ev.make_event(str(kw), start_step=start, duration_steps=int(rng.integers(6, 12)))
        if e:
            events.append(e)
    return events


def main():
    rng = np.random.default_rng(C.SEED)
    links, edge_index = road_network.load()
    n = len(links)
    total_steps = C.N_DAYS * C.STEPS_PER_DAY
    timestamps_idx = np.arange(total_steps)
    print(f"Generating {total_steps} timesteps x {n} links "
          f"({C.N_DAYS} days @ {C.TIME_STEP_MIN} min) ...")

    weather_arr = gen.sample_weather(timestamps_idx, rng)
    events = sample_history_events(rng, total_steps)
    print(f"  weather rain-steps: {(weather_arr > 0).sum()} | events: {len(events)}")

    event_int = ev.event_intensity(events, links, timestamps_idx)
    speed_ratio = gen.compute_speed_ratio(
        links, edge_index, timestamps_idx, weather_arr, event_int, rng, with_noise=True
    )
    X = gen.assemble_features(links, timestamps_idx, weather_arr, event_int, speed_ratio)

    np.savez_compressed(
        C.TRAFFIC_PATH,
        X=X.astype(np.float32),
        speed_ratio=speed_ratio.astype(np.float32),
        weather=weather_arr.astype(np.float32),
        feature_names=np.array(gen.FEATURE_NAMES),
    )
    print(f"  mean speed-ratio {speed_ratio.mean():.3f} "
          f"(1.0=free flow, 0.1=gridlock)")
    print(f"Saved traffic series -> {C.TRAFFIC_PATH}  X={X.shape}")


if __name__ == "__main__":
    main()
