"""Project paths and shared constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA = DATA_DIR / "train.csv"

MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

TARGET = "Time_taken_min"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Coordinates below this absolute value are placeholder/corrupt values in this
# dataset (a chunk of rows carry 0.0 or near-zero lat/lon), which produce
# impossible haversine distances of several thousand kilometres.
MIN_VALID_COORD = 1.0

# Deliveries in this dataset are last-mile; anything beyond this is a data
# error rather than a real trip.
MAX_VALID_DISTANCE_KM = 100.0
