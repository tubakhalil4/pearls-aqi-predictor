
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "historical_raw_backfill.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "historical_features.csv"


RENAME_MAP = {
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_direction",
    "pressure_msl": "pressure",
    "precipitation": "rain",
    "us_aqi": "aqi",
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["time"] = pd.to_datetime(df["time"], utc=True)

    df = df.rename(columns=RENAME_MAP)

    df = df.sort_values(["city", "time"]).reset_index(drop=True)

    # Time-based features
    df["hour"] = df["time"].dt.hour.astype("int32")
    df["day"] = df["time"].dt.day.astype("int32")
    df["month"] = df["time"].dt.month.astype("int32")
    df["day_of_week"] = df["time"].dt.dayofweek.astype("int32")

    # Derived AQI features
    df["aqi_change_rate"] = (
        df.groupby("city")["aqi"]
        .diff()
        .fillna(0)
    )

    df["aqi_rolling_6h"] = (
        df.groupby("city")["aqi"]
        .transform(lambda s: s.rolling(window=6, min_periods=1).mean())
    )

    # Production column order
    columns = [
        "city",
        "time",
        "pm2_5",
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "ozone",
        "sulphur_dioxide",
        "temperature",
        "humidity",
        "wind_speed",
        "wind_direction",
        "pressure",
        "rain",
        "hour",
        "day",
        "month",
        "day_of_week",
        "aqi_change_rate",
        "aqi_rolling_6h",
        "aqi",
    ]

    missing_columns = [c for c in columns if c not in df.columns]

    if missing_columns:
        raise RuntimeError(
            f"Missing required feature columns: {missing_columns}"
        )

    df = df[columns]

    # Remove invalid numeric values
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    df[numeric_columns] = df[numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if df[numeric_columns].isna().any().any():
        raise RuntimeError(
            "Feature dataset contains missing or infinite numeric values."
        )

    return df


def validate_features(df: pd.DataFrame) -> None:
    expected_columns = [
        "city",
        "time",
        "pm2_5",
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "ozone",
        "sulphur_dioxide",
        "temperature",
        "humidity",
        "wind_speed",
        "wind_direction",
        "pressure",
        "rain",
        "hour",
        "day",
        "month",
        "day_of_week",
        "aqi_change_rate",
        "aqi_rolling_6h",
        "aqi",
    ]

    if list(df.columns) != expected_columns:
        raise RuntimeError(
            "Feature columns do not match the production schema."
        )

    if df.empty:
        raise RuntimeError("Historical feature dataset is empty.")

    if df["city"].nunique() != 4:
        raise RuntimeError(
            f"Expected 4 cities, found {df['city'].nunique()}."
        )

    duplicate_count = df.duplicated(
        subset=["city", "time"]
    ).sum()

    if duplicate_count != 0:
        raise RuntimeError(
            f"Found {duplicate_count} duplicate city/time rows."
        )

    if df.isna().any().any():
        raise RuntimeError("Historical feature dataset contains missing values.")


def main() -> None:
    print("=" * 70)
    print("BUILDING HISTORICAL PRODUCTION FEATURES")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Historical raw dataset not found: {INPUT_FILE}"
        )

    print(f"Input:  {INPUT_FILE}")

    raw = pd.read_csv(INPUT_FILE)

    print(f"Raw shape: {raw.shape}")

    features = build_features(raw)

    validate_features(features)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    features.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 70)
    print("HISTORICAL FEATURE DATASET")
    print("=" * 70)

    print(f"Shape: {features.shape}")

    print("\nColumns:")
    for i, column in enumerate(features.columns, start=1):
        print(f"{i:2}. {column}")

    print("\nRows per city:")
    print(features["city"].value_counts().sort_index())

    print("\nMissing values:")
    print(features.isna().sum())

    print("\nTime range:")
    print(features["time"].min())
    print(features["time"].max())

    print("\nDuplicate city/time rows:")
    print(
        features.duplicated(
            subset=["city", "time"]
        ).sum()
    )

    print("\n" + "=" * 70)
    print("HISTORICAL FEATURE BUILD COMPLETE")
    print("=" * 70)

    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
