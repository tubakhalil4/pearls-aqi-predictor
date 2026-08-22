import os
from pathlib import Path

import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry


START_DATE = "2024-01-01"
END_DATE = "2025-12-31"

CITIES = {
    "Islamabad": (33.6844, 73.0479),
    "Karachi": (24.8607, 67.0011),
    "Lahore": (31.5204, 74.3587),
    "Peshawar": (34.0151, 71.5249),
}

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

OUTPUT = Path("data/historical_raw_backfill.csv")


def create_clients():
    cache_session = requests_cache.CachedSession(
        ".cache",
        expire_after=-1,
    )
    retry_session = retry(
        cache_session,
        retries=5,
        backoff_factor=0.2,
    )
    return (
        openmeteo_requests.Client(session=retry_session),
        retry_session,
    )


def fetch_city(client, city, latitude, longitude):
    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "wind_direction_10m",
            "pressure_msl",
        ],
        "timezone": "UTC",
    }

    air_params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": [
            "pm2_5",
            "pm10",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "ozone",
            "sulphur_dioxide",
            "us_aqi",
        ],
        "timezone": "UTC",
    }

    weather_response = client.weather_api(
        WEATHER_URL,
        params=weather_params,
    )[0]

    air_response = client.weather_api(
        AIR_URL,
        params=air_params,
    )[0]

    weather = weather_response.Hourly()
    air = air_response.Hourly()

    weather_times = pd.date_range(
        start=pd.to_datetime(weather.Time(), unit="s", utc=True),
        end=pd.to_datetime(weather.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=weather.Interval()),
        inclusive="left",
    )

    air_times = pd.date_range(
        start=pd.to_datetime(air.Time(), unit="s", utc=True),
        end=pd.to_datetime(air.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=air.Interval()),
        inclusive="left",
    )

    weather_df = pd.DataFrame(
        {
            "time": weather_times,
            "temperature_2m": weather.Variables(0).ValuesAsNumpy(),
            "relative_humidity_2m": weather.Variables(1).ValuesAsNumpy(),
            "precipitation": weather.Variables(2).ValuesAsNumpy(),
            "wind_speed_10m": weather.Variables(3).ValuesAsNumpy(),
            "wind_direction_10m": weather.Variables(4).ValuesAsNumpy(),
            "pressure_msl": weather.Variables(5).ValuesAsNumpy(),
        }
    )

    air_df = pd.DataFrame(
        {
            "time": air_times,
            "pm2_5": air.Variables(0).ValuesAsNumpy(),
            "pm10": air.Variables(1).ValuesAsNumpy(),
            "carbon_monoxide": air.Variables(2).ValuesAsNumpy(),
            "nitrogen_dioxide": air.Variables(3).ValuesAsNumpy(),
            "ozone": air.Variables(4).ValuesAsNumpy(),
            "sulphur_dioxide": air.Variables(5).ValuesAsNumpy(),
            "us_aqi": air.Variables(6).ValuesAsNumpy(),
        }
    )

    df = pd.merge(weather_df, air_df, on="time", how="inner")
    df["city"] = city

    columns = [
        "pm2_5",
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "ozone",
        "sulphur_dioxide",
        "us_aqi",
        "time",
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "wind_speed_10m",
        "wind_direction_10m",
        "pressure_msl",
        "city",
    ]

    return df[columns]


def main():
    client, _ = create_clients()

    frames = []

    for city, (latitude, longitude) in CITIES.items():
        print("-" * 70)
        print(f"FETCHING: {city}")
        print("-" * 70)

        df = fetch_city(
            client,
            city,
            latitude,
            longitude,
        )

        print(f"Rows received: {len(df)}")
        print(f"Time range: {df['time'].min()} to {df['time'].max()}")

        frames.append(df)

    historical = pd.concat(
        frames,
        ignore_index=True,
    )

    historical = historical.sort_values(
        ["city", "time"]
    ).reset_index(drop=True)

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    historical.to_csv(
        OUTPUT,
        index=False,
    )

    print("=" * 70)
    print("HISTORICAL RAW DATA SUMMARY")
    print("=" * 70)
    print(f"Shape: {historical.shape}")
    print("\nRows per city:")
    print(historical["city"].value_counts())
    print("\nTime range:")
    print(historical["time"].min())
    print(historical["time"].max())
    print("\nMissing values:")
    print(historical.isna().sum())
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
