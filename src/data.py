"""Loading and cleaning of the raw food delivery dataset.

The raw CSV is messy in several specific ways, each handled explicitly below:

* Numeric columns arrive as strings, with the literal text ``"NaN"`` standing in
  for missing values.
* ``Time_taken(min)`` is prefixed with the text ``"(min) "``.
* ``Weatherconditions`` is prefixed with the text ``"conditions "``.
* Several categorical columns carry stray trailing whitespace, which would
  otherwise produce duplicate one-hot columns.
* ``Order_Date`` is day-first (``19-03-2022``).
"""

from __future__ import annotations

import pandas as pd

from . import config

# Columns that are strings in the CSV but should be numeric.
_NUMERIC_COLUMNS = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "multiple_deliveries",
]

# Columns whose values need whitespace stripped before use.
_STRING_COLUMNS = [
    "Weatherconditions",
    "Road_traffic_density",
    "Type_of_order",
    "Type_of_vehicle",
    "Festival",
    "City",
]


def load_raw(path=None) -> pd.DataFrame:
    """Read the raw CSV with no transformation beyond parsing."""
    return pd.read_csv(path or config.RAW_DATA)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw dataframe.

    Missing values are left as ``NaN`` rather than imputed: the tree models used
    here handle them natively, and imputing would invent delivery-person ages
    and ratings that were never recorded.
    """
    df = df.copy()

    # Strip whitespace everywhere, then turn the literal string "NaN" into a
    # real missing value. This must happen BEFORE any encoding, otherwise "NaN"
    # survives as its own category.
    for column in df.select_dtypes(include="object"):
        df[column] = df[column].str.strip()
    df = df.replace({"NaN": None, "": None})

    # "conditions Sunny" -> "Sunny"
    df["Weatherconditions"] = df["Weatherconditions"].str.replace(
        "conditions ", "", regex=False
    )

    # "(min) 24" -> 24
    df["Time_taken_min"] = (
        df["Time_taken(min)"].str.replace("(min)", "", regex=False).str.strip().astype(float)
    )
    df = df.drop(columns=["Time_taken(min)"])

    for column in _NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Day-first format. Parsing this as "%Y-%m-%d" silently coerces every single
    # row to NaT, which is what happened in the original version of this project.
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], format="%d-%m-%Y", errors="coerce")

    for column in ("Time_Orderd", "Time_Order_picked"):
        df[column] = _parse_clock_to_minutes(df[column])

    df = df.dropna(subset=["Time_taken_min"])
    return df.reset_index(drop=True)


def _parse_clock_to_minutes(series: pd.Series) -> pd.Series:
    """Convert an ``HH:MM:SS`` column into minutes since midnight.

    Kept as a plain integer count rather than a datetime so that arithmetic on
    it stays meaningful. (Feeding minute counts back into ``pd.to_datetime``
    reinterprets them as nanoseconds since the epoch, collapsing every row to
    hour zero.)
    """
    parsed = pd.to_datetime(series, format="%H:%M:%S", errors="coerce")
    return parsed.dt.hour * 60 + parsed.dt.minute


def load_clean(path=None) -> pd.DataFrame:
    """Convenience wrapper: read the raw CSV and clean it."""
    return clean(load_raw(path))
