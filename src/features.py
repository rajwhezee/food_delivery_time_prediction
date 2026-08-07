"""Feature engineering.

Everything here is computable at the moment the order is placed and picked up,
so nothing leaks information about how long the delivery actually took.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

MINUTES_PER_DAY = 24 * 60

# One-hot encoded (no meaningful order between categories).
_NOMINAL_COLUMNS = ["Type_of_vehicle", "Weatherconditions", "Type_of_order", "City"]

# Ordinal: congestion has a natural ranking, so a single ordered column is a
# better fit for a tree model than four separate one-hot splits.
_TRAFFIC_ORDER = {"Low": 0, "Medium": 1, "High": 2, "Jam": 3}

_DROP_COLUMNS = [
    "ID",
    "Delivery_person_ID",
    "Order_Date",
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
]


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometres, vectorised over numpy arrays."""
    earth_radius_km = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return earth_radius_km * 2 * np.arcsin(np.sqrt(a))


def add_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``distance_km``, blanking out physically impossible values.

    A few thousand rows carry placeholder coordinates at or near (0, 0), which
    yield distances of many thousands of kilometres for what is a last-mile
    delivery. Those are marked missing rather than kept as extreme outliers.
    """
    df = df.copy()

    coordinate_columns = [
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
    ]
    # Some rows store coordinates with a flipped sign; magnitude is correct.
    coords = {c: df[c].abs() for c in coordinate_columns}

    df["distance_km"] = haversine_km(
        coords["Restaurant_latitude"],
        coords["Restaurant_longitude"],
        coords["Delivery_location_latitude"],
        coords["Delivery_location_longitude"],
    )

    invalid = np.zeros(len(df), dtype=bool)
    for column in coordinate_columns:
        invalid |= coords[column] < config.MIN_VALID_COORD
    invalid |= df["distance_km"] > config.MAX_VALID_DISTANCE_KM

    df.loc[invalid, "distance_km"] = np.nan
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add order-timing features derived from the parsed clock columns."""
    df = df.copy()

    # Modulo the day length so an order placed at 23:55 and picked up at 00:10
    # reads as 15 minutes, not -1425.
    df["order_to_pickup_min"] = (
        df["Time_Order_picked"] - df["Time_Orderd"]
    ) % MINUTES_PER_DAY

    df["order_hour"] = df["Time_Orderd"] // 60
    df["is_peak_hour"] = df["order_hour"].isin([12, 13, 19, 20, 21]).astype(int)

    df["day_of_week"] = df["Order_Date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categoricals: ordinal for traffic and festival, one-hot for the rest."""
    df = df.copy()

    df["Road_traffic_density"] = df["Road_traffic_density"].map(_TRAFFIC_ORDER)
    df["Festival"] = df["Festival"].map({"No": 0, "Yes": 1})

    return pd.get_dummies(df, columns=_NOMINAL_COLUMNS, drop_first=True, dtype=float)


def build(df: pd.DataFrame):
    """Run the full pipeline and return the feature matrix and target vector."""
    df = add_distance(df)
    df = add_time_features(df)
    df = encode(df)
    df = df.drop(columns=[c for c in _DROP_COLUMNS if c in df.columns])

    y = df[config.TARGET]
    X = df.drop(columns=[config.TARGET])
    return X, y
