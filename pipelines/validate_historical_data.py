from pathlib import Path

import pandas as pd


INPUT = Path("data/historical_raw_backfill.csv")

EXPECTED_CITIES = {
    "Islamabad",
    "Karachi",
    "Lahore",
    "Peshawar",
}


def main():
    print("=" * 70)
    print("VALIDATING HISTORICAL BACKFILL")
    print("=" * 70)

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Historical dataset not found: {INPUT}"
        )

    df = pd.read_csv(
        INPUT,
        parse_dates=["time"],
    )

    required_columns = {
        "city",
        "time",
        "pm2_5",
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "ozone",
        "sulphur_dioxide",
        "us_aqi",
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "wind_speed_10m",
        "wind_direction_10m",
        "pressure_msl",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise RuntimeError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    if set(df["city"].unique()) != EXPECTED_CITIES:
        raise RuntimeError(
            f"Unexpected cities: {sorted(df['city'].unique())}"
        )

    missing = int(df.isna().sum().sum())

    duplicates = int(
        df.duplicated(
            subset=["city", "time"]
        ).sum()
    )

    continuity_errors = {}

    for city, group in df.groupby("city"):
        times = (
            pd.to_datetime(group["time"], utc=True)
            .sort_values()
        )

        expected = pd.date_range(
            times.min(),
            times.max(),
            freq="h",
        )

        continuity_errors[city] = len(
            expected.difference(times)
        )

    print(f"Shape: {df.shape}")
    print("\nRows per city:")
    print(df["city"].value_counts())
    print("\nTime range:")
    print(df["time"].min())
    print(df["time"].max())
    print(f"\nMissing values: {missing}")
    print(f"Duplicate city/time rows: {duplicates}")
    print(f"Hourly continuity errors: {continuity_errors}")

    if missing != 0:
        raise RuntimeError("Missing values detected.")

    if duplicates != 0:
        raise RuntimeError("Duplicate city/time rows detected.")

    if any(continuity_errors.values()):
        raise RuntimeError(
            "Hourly continuity errors detected."
        )

    print("\n" + "=" * 70)
    print("HISTORICAL BACKFILL VALIDATION PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
