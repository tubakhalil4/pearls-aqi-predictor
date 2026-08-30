import os
from pathlib import Path
import joblib
import hopsworks
import numpy as np
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
import xgboost as xgb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_ROOT / "forecast_3days.csv"


CITIES = {
    "Islamabad": (33.6844, 73.0479),
    "Karachi": (24.8607, 67.0011),
    "Lahore": (31.5204, 74.3587),
    "Peshawar": (34.0151, 71.5249),
}

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

FEATURE_GROUP_NAME = "aqi_features_v2"
FEATURE_GROUP_VERSION = 1

MODEL_NAMES = {
    "24h": "aqi_forecast_24h",
    "48h": "aqi_forecast_48h",
    "72h": "aqi_forecast_72h",
}

FEATURE_COLUMNS = [
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
]



def write_current_features_to_feature_store(
    feature_store,
    feature_rows,
):
    """
    Persist fresh hourly Open-Meteo feature rows into
    the existing Hopsworks Feature Group.

    Feature Group:
        aqi_features_v2, version 1

    Primary key:
        city + time

    Event time:
        time
    """

    print("\n" + "=" * 70)
    print("WRITING FRESH FEATURES TO HOPSWORKS FEATURE STORE")
    print("=" * 70)

    if feature_rows is None or feature_rows.empty:
        raise RuntimeError(
            "No feature rows available for Hopsworks write."
        )

    fg = feature_store.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
    )

    # ------------------------------------------------
    # Always work from feature_rows.
    # NEVER reference write_df here.
    # ------------------------------------------------

    write_df = feature_rows.copy()

    # ------------------------------------------------
    # Ensure event time is UTC-aware.
    # ------------------------------------------------

    if "time" not in write_df.columns:
        raise RuntimeError(
            "Feature Store write requires a 'time' column."
        )

    write_df["time"] = pd.to_datetime(
        write_df["time"],
        utc=True,
    )

    # ------------------------------------------------
    # Production Feature Group schema.
    # ------------------------------------------------

    required_columns = [
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

    # ------------------------------------------------
    # Protect primary key.
    # Hopsworks primary key = city + time
    # ------------------------------------------------

    if "city" not in write_df.columns:
        raise RuntimeError(
            "Feature Store write could not determine city column."
        )

    write_df["city"] = (
        write_df["city"]
        .astype(str)
        .str.strip()
    )

    if write_df["city"].eq("").any():
        raise RuntimeError(
            "Feature Store write contains an empty city value."
        )

    # ------------------------------------------------
    # Validate schema BEFORE selecting columns.
    # ------------------------------------------------

    missing = [
        col
        for col in required_columns
        if col not in write_df.columns
    ]

    if missing:
        raise RuntimeError(
            "Feature Store write schema mismatch. "
            f"Missing columns: {missing}"
        )

    # ------------------------------------------------
    # Keep exactly production schema/order.
    # ------------------------------------------------

    write_df = write_df[required_columns].copy()

    # ------------------------------------------------
    # Remove duplicate primary keys.
    # ------------------------------------------------

    write_df = (
        write_df
        .drop_duplicates(
            subset=["city", "time"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    if write_df.empty:
        raise RuntimeError(
            "No rows remain after primary-key deduplication."
        )

    # ------------------------------------------------
    # Final primary-key validation.
    # ------------------------------------------------

    if write_df[["city", "time"]].isna().any().any():
        raise RuntimeError(
            "Feature Store primary key contains NULL values."
        )

    duplicate_keys = write_df.duplicated(
        subset=["city", "time"]
    ).sum()

    if duplicate_keys:
        raise RuntimeError(
            f"Feature Store write still contains "
            f"{duplicate_keys} duplicate primary key(s)."
        )

    # ------------------------------------------------
    # Logging.
    # ------------------------------------------------

    print(
        f"Rows being written: {len(write_df)}"
    )

    print(
        "Latest rows:\n"
        + write_df[
            [
                "city",
                "time",
                "aqi",
                "aqi_change_rate",
                "aqi_rolling_6h",
            ]
        ].to_string(index=False)
    )

    # ------------------------------------------------
    # Hopsworks insert.
    # ------------------------------------------------

    fg.insert(
        write_df,
        write_options={
            "wait_for_job": True,
        },
    )

    print(
        f"Successfully wrote "
        f"{len(write_df)} row(s) to "
        f"{FEATURE_GROUP_NAME} v{FEATURE_GROUP_VERSION}"
    )

    return write_df


def create_api_client():
    cache_session = requests_cache.CachedSession(
        ".cache",
        expire_after=300,
    )

    retry_session = retry(
        cache_session,
        retries=5,
        backoff_factor=0.2,
    )

    return openmeteo_requests.Client(
        session=retry_session
    )


def fetch_open_meteo(city, latitude, longitude):
    print(f"\nFetching API data: {city}")

    client = create_api_client()

    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "wind_direction_10m",
            "pressure_msl",
        ],
        "forecast_days": 4,
        "timezone": "UTC",
    }

    air_params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": [
            "pm2_5",
            "pm10",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "ozone",
            "sulphur_dioxide",
            "us_aqi",
        ],
        "forecast_days": 4,
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
        start=pd.to_datetime(
            weather.Time(),
            unit="s",
            utc=True,
        ),
        end=pd.to_datetime(
            weather.TimeEnd(),
            unit="s",
            utc=True,
        ),
        freq=pd.Timedelta(
            seconds=weather.Interval()
        ),
        inclusive="left",
    )

    air_times = pd.date_range(
        start=pd.to_datetime(
            air.Time(),
            unit="s",
            utc=True,
        ),
        end=pd.to_datetime(
            air.TimeEnd(),
            unit="s",
            utc=True,
        ),
        freq=pd.Timedelta(
            seconds=air.Interval()
        ),
        inclusive="left",
    )

    weather_df = pd.DataFrame({
        "time": weather_times,
        "temperature": weather.Variables(0).ValuesAsNumpy(),
        "humidity": weather.Variables(1).ValuesAsNumpy(),
        "rain": weather.Variables(2).ValuesAsNumpy(),
        "wind_speed": weather.Variables(3).ValuesAsNumpy(),
        "wind_direction": weather.Variables(4).ValuesAsNumpy(),
        "pressure": weather.Variables(5).ValuesAsNumpy(),
    })

    air_df = pd.DataFrame({
        "time": air_times,
        "pm2_5": air.Variables(0).ValuesAsNumpy(),
        "pm10": air.Variables(1).ValuesAsNumpy(),
        "carbon_monoxide": air.Variables(2).ValuesAsNumpy(),
        "nitrogen_dioxide": air.Variables(3).ValuesAsNumpy(),
        "ozone": air.Variables(4).ValuesAsNumpy(),
        "sulphur_dioxide": air.Variables(5).ValuesAsNumpy(),
        "aqi": air.Variables(6).ValuesAsNumpy(),
    })

    return pd.merge(
        weather_df,
        air_df,
        on="time",
        how="inner",
    ).sort_values("time").reset_index(drop=True)


def prepare_future_features(raw_future, latest_state):
    df = raw_future.copy()

    df["hour"] = df["time"].dt.hour.astype("int64")
    df["day"] = df["time"].dt.day.astype("int64")
    df["month"] = df["time"].dt.month.astype("int64")
    df["day_of_week"] = (
        df["time"].dt.dayofweek.astype("int64")
    )

    current_change = float(
        latest_state["aqi_change_rate"]
    )

    current_rolling = float(
        latest_state["aqi_rolling_6h"]
    )

    df["aqi_change_rate"] = current_change
    df["aqi_rolling_6h"] = current_rolling

    return df


def select_forecast_rows(
    future,
    forecast_origin,
):
    targets = [
        forecast_origin + pd.Timedelta(hours=24),
        forecast_origin + pd.Timedelta(hours=48),
        forecast_origin + pd.Timedelta(hours=72),
    ]

    selected = []

    for target in targets:
        exact = future[
            future["time"] == target
        ]

        if exact.empty:
            candidates = future[
                future["time"] > target
            ]

            if candidates.empty:
                raise RuntimeError(
                    f"No API row available for {target}"
                )

            row = candidates.iloc[0]
        else:
            row = exact.iloc[0]

        selected.append(row)

    return selected




def load_models(model_registry):
    models = {}

    for horizon, model_name in MODEL_NAMES.items():

        print(f"\nLoading Model Registry model: {model_name}")

        model = model_registry.get_best_model(
            model_name,
            metric="rmse",
            direction="min",
        )

        local_dir = model.download()

        pkl_path = Path(local_dir) / "model.pkl"

        if not pkl_path.exists():
            raise RuntimeError(
                f"No model.pkl artifact found for {model_name}"
            )

        booster = joblib.load(pkl_path)
        models[horizon] = booster

        feature_columns_path = Path(local_dir) / "feature_columns.json"
        if feature_columns_path.exists():
            saved_features = pd.read_json(
                feature_columns_path, typ="series"
            ).tolist()
            if list(saved_features) != FEATURE_COLUMNS:
                raise RuntimeError(
                    f"Feature schema mismatch for {model_name}"
                )

        print("Feature schema: OK")

    return models

def write_all_features_to_feature_store(
    feature_store,
    feature_rows,
):
    """
    Persist fresh hourly Open-Meteo feature rows for ALL cities
    into the existing Hopsworks Feature Group in a single insert.

    Batching all cities into one fg.insert() call means only one
    materialization job is launched per hourly run, instead of
    one per city -- this avoids queuing multiple jobs back-to-back
    on Hopsworks' shared free-tier job execution slot.

    Feature Group:
        aqi_features_v2, version 1

    Primary key:
        city + time

    Event time:
        time
    """

    print("\n" + "=" * 70)
    print("WRITING FRESH FEATURES TO HOPSWORKS FEATURE STORE")
    print("=" * 70)

    if feature_rows is None or feature_rows.empty:
        raise RuntimeError(
            "No feature rows available for Hopsworks write."
        )

    fg = feature_store.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
    )

    write_df = feature_rows.copy()

    # Ensure event time is UTC-aware.
    write_df["time"] = pd.to_datetime(
        write_df["time"],
        utc=True,
    )

    if "city" not in write_df.columns:
        raise RuntimeError(
            "Feature Store write could not determine city column. "
            "Ensure 'city' is attached to each row before calling "
            "write_all_features_to_feature_store()."
        )

    write_df["city"] = write_df["city"].astype(str)

    # The production Feature Group schema.
    required_columns = [
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

    missing = [
        col
        for col in required_columns
        if col not in write_df.columns
    ]

    if missing:
        raise RuntimeError(
            "Feature Store write schema mismatch. "
            f"Missing columns: {missing}"
        )

    # Keep exactly the production schema and order.
    write_df = write_df[required_columns].copy()

    # Hopsworks feature group columns are 'double' (float64).
    # Open-Meteo returns float32 arrays, which Hopsworks' schema
    # check rejects even though the values are compatible.
    float_columns = [
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
        "aqi",
    ]
    write_df[float_columns] = write_df[float_columns].astype("float64")

    # Remove accidental duplicate primary keys.
    write_df = (
        write_df
        .drop_duplicates(
            subset=["city", "time"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    print(
        f"Rows being written: {len(write_df)}"
    )

    print(
        "Latest rows:\n"
        + write_df[
            [
                "city",
                "time",
                "aqi",
                "aqi_change_rate",
                "aqi_rolling_6h",
            ]
        ].to_string(index=False)
    )

    # Single insert for all cities -> single materialization job.
    fg.insert(
        write_df,
        write_options={
            "wait_for_job": True,
        },
    )

    print(
        f"✅ Successfully wrote "
        f"{len(write_df)} row(s) to "
        f"{FEATURE_GROUP_NAME} v{FEATURE_GROUP_VERSION}"
    )

    return write_df


def read_feature_store(feature_store):
    print("\n" + "=" * 70)
    print("READING HOPSWORKS FEATURE STORE")
    print("=" * 70)

    fg = feature_store.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
    )

    feature_df = fg.read()

    feature_df["time"] = pd.to_datetime(
        feature_df["time"],
        utc=True,
    )

    feature_df = feature_df.sort_values(
        ["city", "time"]
    )

    latest_states = (
        feature_df
        .groupby("city", as_index=False)
        .tail(1)
        .copy()
    )

    print(
        f"Feature group: {FEATURE_GROUP_NAME}"
    )
    print(
        f"Version: {FEATURE_GROUP_VERSION}"
    )
    print(
        f"Rows read: {len(feature_df)}"
    )

    print("\nLatest Feature Store state:")
    print(
        latest_states[
            [
                "city",
                "time",
                "aqi",
                "aqi_change_rate",
                "aqi_rolling_6h",
            ]
        ].to_string(index=False)
    )

    return feature_df, latest_states


def main():

    print("=" * 70)
    print("PEARLS AQI PREDICTOR — HOURLY PRODUCTION FORECAST")
    print("=" * 70)

    # ------------------------------------------------------------
    # HOPSWORKS
    # ------------------------------------------------------------

    print("=" * 70)
    print("CONNECTING TO HOPSWORKS")
    print("=" * 70)

    project = hopsworks.login()

    print("Project:", project.name)

    feature_store = project.get_feature_store()
    model_registry = project.get_model_registry()

    print("Feature Store connected")
    print("Model Registry connected")

    # ------------------------------------------------------------
    # FEATURE STORE
    # ------------------------------------------------------------

    feature_df, latest_states = read_feature_store(
        feature_store
    )

    # ------------------------------------------------------------
    # MODELS
    # ------------------------------------------------------------

    models = load_models(
        model_registry
    )

    output_rows = []
    feature_rows_all = []

    # ------------------------------------------------------------
    # CURRENT PRODUCTION FORECAST
    # ------------------------------------------------------------

    for city, (latitude, longitude) in CITIES.items():

        city_state = latest_states[
            latest_states["city"] == city
        ]

        if city_state.empty:
            raise RuntimeError(
                f"No current Feature Store state for {city}"
            )

        latest_state = city_state.iloc[0]

        raw_future = fetch_open_meteo(
            city,
            latitude,
            longitude,
        )

        api_current = raw_future.iloc[0]

        current_aqi = float(
            api_current["aqi"]
        )

        forecast_origin = pd.to_datetime(
            api_current["time"],
            utc=True,
        )

        current_rows = raw_future[
            raw_future["time"] <= forecast_origin
        ].copy()

        if len(current_rows) >= 2:
            previous_aqi = float(
                current_rows.iloc[-2]["aqi"]
            )

            current_change_rate = (
                current_aqi - previous_aqi
            )
        else:
            current_change_rate = float(
                latest_state["aqi_change_rate"]
            )

        rolling_rows = current_rows.tail(6)

        if not rolling_rows.empty:
            current_rolling_6h = float(
                rolling_rows["aqi"].mean()
            )
        else:
            current_rolling_6h = float(
                latest_state["aqi_rolling_6h"]
            )

        print("\n" + "=" * 70)
        print(city)
        print("=" * 70)
        print(
            "Forecast origin:",
            forecast_origin,
        )
        print(
            "Current AQI:",
            current_aqi,
        )
        print(
            "Current AQI change rate:",
            current_change_rate,
        )
        print(
            "Current AQI rolling 6h:",
            current_rolling_6h,
        )

        future = prepare_future_features(
            raw_future,
            latest_state,
        )

        current_api_rows = future.copy()

        if current_api_rows.empty:
            raise RuntimeError(
                f"No API feature rows available for {city}"
            )

        current_api_time = pd.to_datetime(
            current_api_rows["time"],
            utc=True,
        ).max()

        current_api_row = (
            current_api_rows[
                current_api_rows["time"] == current_api_time
            ]
            .copy()
        )

        if current_api_row.empty:
            raise RuntimeError(
                f"Could not identify current API row for {city}"
            )

        # Attach city, then queue this row for a single combined
        # write after the loop finishes for all cities.
        current_api_row.insert(0, "city", city)
        feature_rows_all.append(current_api_row)

        future = future[
            future["time"] > forecast_origin
        ].copy()

        future["aqi_change_rate"] = (
            current_change_rate
        )

        future["aqi_rolling_6h"] = (
            current_rolling_6h
        )

        if future.empty:
            raise RuntimeError(
                f"No future API rows available for {city}"
            )

        selected = select_forecast_rows(
            future,
            forecast_origin,
        )

        predictions = {}

        for horizon, row in zip(
            ["24h", "48h", "72h"],
            selected,
        ):

            X = pd.DataFrame(
                [row[FEATURE_COLUMNS].to_dict()]
            )

            X = X[
                FEATURE_COLUMNS
            ]

            prediction = float(
                models[horizon].predict(X)[0]
            )

            prediction = max(
                0.0,
                prediction,
            )

            predictions[horizon] = prediction

            print(
                f"{horizon} prediction: "
                f"{prediction:.2f}"
            )

        output_rows.append(
            {
                "city": city,
                "forecast_origin": forecast_origin,
                "current_aqi": current_aqi,
                "aqi_24h": predictions["24h"],
                "aqi_48h": predictions["48h"],
                "aqi_72h": predictions["72h"],
                "forecast_24h_time": (
                    forecast_origin
                    + pd.Timedelta(hours=24)
                ),
                "forecast_48h_time": (
                    forecast_origin
                    + pd.Timedelta(hours=48)
                ),
                "forecast_72h_time": (
                    forecast_origin
                    + pd.Timedelta(hours=72)
                ),
            }
        )

    # ------------------------------------------------------------
    # SINGLE COMBINED FEATURE STORE WRITE (all 4 cities at once)
    # ------------------------------------------------------------

    combined_features = pd.concat(
        feature_rows_all,
        ignore_index=True,
    )

    write_all_features_to_feature_store(
        feature_store,
        combined_features,
    )

    # ------------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------------

    forecast_df = pd.DataFrame(
        output_rows
    )

    forecast_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 70)
    print("FINAL HOURLY FORECAST")
    print("=" * 70)

    print(
        forecast_df.to_string(index=False)
    )

    print(
        f"\nOutput: {OUTPUT_FILE}"
    )

    print("\n" + "=" * 70)
    print("✅ HOURLY PRODUCTION FORECAST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
