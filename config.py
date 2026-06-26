"""
Central configuration for the Bangalore Traffic Flow Prediction System.

Everything tunable lives here so the rest of the codebase stays clean.
Coordinates are unprojected latitude-longitude (EPSG:4326).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

LINKS_PATH = DATA_DIR / "links.pkl"          # GeoDataFrame of road segments
EDGE_INDEX_PATH = DATA_DIR / "edge_index.npy"  # link adjacency for the GNN
META_PATH = DATA_DIR / "meta.json"
TRAFFIC_PATH = DATA_DIR / "traffic.npz"      # generated spatio-temporal tensors
MODEL_PATH = MODEL_DIR / "stgnn.pt"

# ---------------------------------------------------------------------------
# Study area: Central / East Bangalore (ORR corridor, Silk Board, Marathahalli)
# bbox order for OSMnx 2.x is (left, bottom, right, top) = (W, S, E, N)
# ---------------------------------------------------------------------------
BBOX = (77.590, 12.900, 77.720, 13.010)
NETWORK_TYPE = "drive"
NETWORK_CENTER = (12.955, 77.655)  # (lat, lon) ~ centre of the box, used for weather

# Only keep arterial roads so the graph stays tractable on a laptop CPU.
MAJOR_ROADS_FILTER = (
    '["highway"~"motorway|trunk|primary|secondary|tertiary|'
    'motorway_link|trunk_link|primary_link|secondary_link"]'
)
CONSOLIDATE_TOLERANCE_M = 15  # merge intersection nodes within this many metres

# Default free-flow speeds (km/h) by OSM road class, used when maxspeed is missing.
DEFAULT_SPEED_KMH = {
    "motorway": 80, "trunk": 60, "primary": 50, "secondary": 45,
    "tertiary": 40, "residential": 30, "unclassified": 35,
    "motorway_link": 50, "trunk_link": 45, "primary_link": 40,
    "secondary_link": 35, "tertiary_link": 35,
}
ROAD_CLASS_ORDER = ["tertiary", "secondary", "primary", "trunk", "motorway"]

# ---------------------------------------------------------------------------
# Known congestion hotspots (lat, lon, severity 0-1).
# Links near these get extra baseline congestion in the generator.
# ---------------------------------------------------------------------------
HOTSPOTS = [
    ("Silk Board Junction",     12.9172, 77.6228, 1.00),
    ("Marathahalli Bridge",     12.9560, 77.7010, 0.95),
    ("KR Puram / Tin Factory",  12.9990, 77.6640, 0.90),
    ("Bellandur ORR",           12.9260, 77.6780, 0.85),
    ("Iblur Junction",          12.9230, 77.6720, 0.80),
    ("Domlur Flyover",          12.9610, 77.6380, 0.70),
    ("Indiranagar 100ft Rd",    12.9719, 77.6412, 0.65),
    ("Koramangala Sony World",  12.9350, 77.6240, 0.70),
]
HOTSPOT_RADIUS_M = 600  # influence radius of a hotspot

# ---------------------------------------------------------------------------
# Event venues, used both for the demo events and the natural-language parser.
# keyword -> (display name, lat, lon, crowd 0-1, radius_m)
# ---------------------------------------------------------------------------
EVENT_VENUES = {
    "chinnaswamy": ("M. Chinnaswamy Stadium", 12.9788, 77.5996, 1.00, 1600),
    "stadium":     ("M. Chinnaswamy Stadium", 12.9788, 77.5996, 1.00, 1600),
    "cricket":     ("M. Chinnaswamy Stadium", 12.9788, 77.5996, 1.00, 1600),
    "indiranagar": ("Indiranagar",            12.9719, 77.6412, 0.70, 1000),
    "concert":     ("Indiranagar",            12.9719, 77.6412, 0.70, 1000),
    "bellandur":   ("Bellandur Tech Park",    12.9260, 77.6780, 0.80, 1300),
    "tech park":   ("Bellandur Tech Park",    12.9260, 77.6780, 0.80, 1300),
    "marathahalli":("Marathahalli",           12.9560, 77.7010, 0.75, 1200),
    "koramangala": ("Koramangala",            12.9350, 77.6240, 0.70, 1100),
}

# ---------------------------------------------------------------------------
# Temporal / model settings
# ---------------------------------------------------------------------------
TIME_STEP_MIN = 15      # data resolution
STEPS_PER_DAY = 24 * 60 // TIME_STEP_MIN  # 96
N_DAYS = 21             # length of generated history
START_DATE = "2025-01-06"  # a Monday, so weekday/weekend line up cleanly

IN_STEPS = 8            # 2 hours of history fed to the model
OUT_STEPS = 4           # forecast horizon = 1 hour ahead (4 x 15 min)

# GNN hyperparameters
HIDDEN = 64
GCN_LAYERS = 2
DROPOUT = 0.1

# Training
EPOCHS = 15             # best-val checkpoint saved each epoch; safe to Ctrl+C early
BATCH_SIZE = 8          # full Bangalore graph is ~2.6k nodes; lower to 4 if OOM
LR = 1e-3
WEIGHT_DECAY = 1e-5
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
SEED = 42

# Weather intensity -> speed penalty multiplier (used by generator & simulator)
# 0 = clear, 1 = light, 2 = moderate, 3 = heavy
WEATHER_PENALTY = {0: 0.00, 1: 0.06, 2: 0.14, 3: 0.26}
WEATHER_LABELS = {0: "Clear", 1: "Light rain", 2: "Moderate rain", 3: "Heavy rain"}