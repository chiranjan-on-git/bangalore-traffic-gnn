# Bangalore Traffic Flow Prediction System

A spatio-temporal **Graph Neural Network** that forecasts road congestion across
the Central / East Bangalore arterial network (Outer Ring Road, Silk Board,
Marathahalli, KR Puram, Bellandur, Indiranagar). It integrates a real road
network, live weather, and an event layer, then serves an interactive
"what-if" simulator as a web app.

```
Real OSM road graph  +  live weather  +  events  ->  ST-GNN  ->  congestion map
```

## What it does

- **Multi-source data integration** - real road network from OpenStreetMap
  (OSMnx), live weather from Open-Meteo (no API key), and a configurable event
  layer (stadiums, tech parks, concerts).
- **Graph Neural Network modelling** - each road *segment* is a node; segments
  that meet at a junction are connected. A stacked GCN mixes each segment with
  its neighbours so congestion can propagate across the network.
- **Time-series forecasting** - a GRU over the graph-embedded sequence predicts
  the next hour of speed in 15-minute steps.
- **Interactive simulation output** - a Streamlit app with a colour-coded map
  (green = free flowing, red = jammed), manual scenario controls, and a
  natural-language query mode ("if it rains during evening rush near
  Marathahalli...").

## A note on data honesty

The road network is **real** and the weather is **live**. Bangalore has no free
public real-time traffic-sensor feed, so the traffic itself is **realistic
synthetic** data, generated from patterns the city genuinely exhibits: twin
weekday peaks, milder weekends, hotspot junctions, rain slowdowns, event surges,
and - importantly - spatial propagation between connected segments, which is
exactly the structure the GNN is built to exploit. The pipeline is designed so
that a real speed feed could be dropped in to replace the generator later.

## Project layout

```
bangalore-traffic-gnn/
├── config.py                 # all settings: bbox, hotspots, hyperparameters
├── requirements.txt
├── app.py                    # Streamlit interactive simulator
├── src/
│   ├── road_network.py       # OSM download -> link graph (OSMnx 2.x)
│   ├── weather.py            # live Open-Meteo fetch
│   ├── events.py             # event layer + spatial/temporal impact
│   ├── traffic_generator.py  # realistic synthetic traffic (core engine)
│   ├── dataset.py            # windowing + train/val/test split + scaling
│   ├── model.py              # the spatio-temporal GNN (GCN + GRU)
│   ├── train.py              # training loop, MAE/RMSE, checkpointing
│   └── simulate.py           # scenario engine + natural-language parser
├── scripts/
│   ├── 01_build_network.py   # downloads the road graph (needs internet)
│   ├── 02_generate_data.py   # generates the traffic history
│   └── 03_train.py           # trains the model
├── data/                     # generated artifacts (gitignored)
└── models/                   # trained checkpoint (gitignored)
```

## Setup (macOS)

```bash
cd bangalore-traffic-gnn
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`torch` and `torch-geometric` install from plain PyPI wheels - no special index
or C++ extension build is needed for this model on macOS (Apple Silicon or
Intel). Training uses the MPS (Apple GPU) backend automatically if available,
otherwise CPU.

## Run it

```bash
# 1. Download the real Bangalore road network (needs internet)
python scripts/01_build_network.py

# 2. Generate the traffic history (no internet needed)
python scripts/02_generate_data.py

# 3. Train the spatio-temporal GNN
python scripts/03_train.py

# 4. Launch the interactive simulator
streamlit run app.py
```

Steps 1-3 take a few minutes in total on a laptop. The app then opens in your
browser at http://localhost:8501.

## Using the app

- **Manual controls** - choose weekday/weekend, time of day, weather, and active
  events; the map updates with the predicted speed on every segment.
- **Natural-language query** - type a scenario in plain English and it is parsed
  into conditions, e.g. *"cricket match at Chinnaswamy with heavy rain during
  evening rush"*.
- **Live weather** - pull Bangalore's current conditions into the scenario.
- **Baseline delta** - toggle to colour roads by how much worse (or better) the
  scenario is versus a clear, event-free baseline.

## Tuning

Everything is in `config.py`: expand `BBOX` for more of the city, adjust
`HOTSPOTS` / `EVENT_VENUES`, change the forecast horizon (`OUT_STEPS`), history
length (`N_DAYS`), or model size (`HIDDEN`, `GCN_LAYERS`).

## How the model works

For each of the past `IN_STEPS` timesteps, the input features (recent speed,
time-of-day encodings, weekend flag, weather, event intensity, and static road
attributes) are passed through stacked `GCNConv` layers over the segment graph.
The resulting per-segment embeddings form a sequence that a GRU consumes to
capture temporal build-up and dissipation of jams. A linear head emits the next
`OUT_STEPS` speed ratios per segment. The whole batch is processed in one GCN
call by treating it as a disjoint union of graph copies.
