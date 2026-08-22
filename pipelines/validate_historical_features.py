
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_FILE = PROJECT_ROOT / "data" / "historical_features.csv"

EXPECTED_COLUMNS = [
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

EXPECTED_CITIES = [
    "Islamabad",
    "Karachi",
    "Lahore",
    "Peshawar",
]


def main() -> None:
    print("=" * 70)
    print("VALIDATING HISTORICAL PRODUCTION FEATURES")
    print("=" * 70)

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {FEATURE_FILE}"
        )

    df = pd.read_csv(FEATURE_FILE)

    print(f"Shape: {df.shape}")

    if list(df.columns) != EXPECTED_COLUMNS:
        raise RuntimeError(
            "Feature columns do not match the expected production schema."
        )

    print("\nColumn schema: ✅")

    cities = sorted(df["city"].dropna().unique().tolist())

    if cities != sorted(EXPECTED_CITIES):
        raise RuntimeError(
            f"Unexpected cities: {cities}"
        )

    print("Cities: ✅")

    if df.isna().any().any():
        missing = df.isna().sum()
        raise RuntimeError(
            f"Missing values detected:\n{missing[missing > 0]}"
        )

    print("Missing values: 0 ✅")

    duplicate_count = df.duplicated(
        subset=["city", "time"]
    ).sum()

    if duplicate_count != 0:
        raise RuntimeError(
            f"Duplicate city/time rows: {duplicate_count}"
        )

    print("Duplicate city/time rows: 0 ✅")

    df["time"] = pd.to_datetime(df["time"], utc=True)

    print("\nRows per city:")
    print(df["city"].value_counts().sort_index())

    print("\nTime range:")
    print(df["time"].min())
    print(df["time"].max())

    print("\nAQI statistics:")
    print(df["aqi"].describe())

    print("\n" + "=" * 70)
    print("✅ HISTORICAL PRODUCTION FEATURES VALIDATION PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
