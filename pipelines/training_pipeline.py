import os
import shutil
from pathlib import Path

import hopsworks
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_GROUP_NAME = "aqi_features_v2"
FEATURE_GROUP_VERSION = 1

MODEL_NAMES = {
    "24h": "aqi_forecast_24h",
    "48h": "aqi_forecast_48h",
    "72h": "aqi_forecast_72h",
}

HORIZON_HOURS = {
    "24h": 24,
    "48h": 48,
    "72h": 72,
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

RAW_FEATURES_FOR_FUTURE_ROW = [
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
]


def load_feature_store_data(feature_store):
    print("\n" + "=" * 70)
    print("READING HOPSWORKS FEATURE STORE FOR TRAINING")
    print("=" * 70)

    fg = feature_store.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
    )

    df = fg.read()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values(["city", "time"]).reset_index(drop=True)

    print(f"Rows read: {len(df)}")
    print(f"Cities: {sorted(df['city'].unique())}")
    print(
        f"Time range: {df['time'].min()} -> {df['time'].max()}"
    )

    return df


def build_supervised_dataset(df, horizon_hours):
    """
    For each origin row at time t0, builds:
      X = weather/pollutant values at time (t0 + horizon_hours),
          plus calendar features for that future time,
          plus aqi_change_rate / aqi_rolling_6h AS KNOWN AT t0
      y = actual AQI at time (t0 + horizon_hours)

    This mirrors exactly what hourly_forecast.py does at inference
    time (future weather/pollutant forecast + current AQI trend
    state), so training distribution matches production distribution.
    """

    all_rows = []

    for city, g in df.groupby("city"):
        g = g.set_index("time").sort_index()

        future_index = g.index + pd.Timedelta(hours=horizon_hours)
        future_rows = g.reindex(future_index)
        future_rows.index = g.index

        combined = pd.DataFrame(index=g.index)

        for col in RAW_FEATURES_FOR_FUTURE_ROW:
            combined[col] = future_rows[col].values

        combined["hour"] = future_index.hour
        combined["day"] = future_index.day
        combined["month"] = future_index.month
        combined["day_of_week"] = future_index.dayofweek

        combined["aqi_change_rate"] = g["aqi_change_rate"].values
        combined["aqi_rolling_6h"] = g["aqi_rolling_6h"].values

        combined["target"] = future_rows["aqi"].values
        combined["city"] = city
        combined["origin_time"] = g.index

        all_rows.append(combined.reset_index(drop=True))

    result = pd.concat(all_rows, ignore_index=True)
    result = result.dropna(subset=["target"] + FEATURE_COLUMNS)

    return result


def chronological_split(dataset, test_fraction=0.2):
    dataset = dataset.sort_values("origin_time").reset_index(drop=True)
    cutoff_idx = int(len(dataset) * (1 - test_fraction))
    cutoff_time = dataset.iloc[cutoff_idx]["origin_time"]

    train = dataset[dataset["origin_time"] < cutoff_time]
    test = dataset[dataset["origin_time"] >= cutoff_time]

    return train, test, cutoff_time


def evaluate(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def train_and_compare(train, test, horizon_label):
    X_train = train[FEATURE_COLUMNS]
    y_train = train["target"]
    X_test = test[FEATURE_COLUMNS]
    y_test = test["target"]

    candidates = {}

    # ---- Statistical model ----
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_metrics = evaluate(y_test, ridge.predict(X_test))
    candidates["ridge"] = (ridge, ridge_metrics)

    # ---- Ensemble tree model ----
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_metrics = evaluate(y_test, rf.predict(X_test))
    candidates["random_forest"] = (rf, rf_metrics)

    # ---- Gradient boosted trees ----
    xgb_model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    xgb_model.fit(X_train, y_train)
    xgb_metrics = evaluate(y_test, xgb_model.predict(X_test))
    candidates["xgboost"] = (xgb_model, xgb_metrics)

    print(f"\n{'=' * 70}")
    print(f"MODEL COMPARISON — {horizon_label}")
    print(f"{'=' * 70}")
    comparison_df = pd.DataFrame(
        {
            name: metrics
            for name, (_, metrics) in candidates.items()
        }
    ).T
    print(comparison_df.to_string())

    best_name = min(
        candidates,
        key=lambda name: candidates[name][1]["rmse"],
    )
    best_model, best_metrics = candidates[best_name]

    print(f"\nBest model for {horizon_label}: {best_name}")
    print(f"Metrics: {best_metrics}")

    return best_name, best_model, best_metrics, comparison_df


def register_model(model_registry, model_name, model_type, model_obj, metrics, feature_columns):
    local_dir = PROJECT_ROOT / f"_tmp_model_{model_name}"
    if local_dir.exists():
        shutil.rmtree(local_dir)
    local_dir.mkdir(parents=True)

    # Save via joblib so any sklearn-API model type can be loaded
    # the same way at inference time.
    joblib.dump(model_obj, local_dir / "model.pkl")

    metadata = {
        "model_type": model_type,
        "feature_columns": feature_columns,
    }
    pd.Series(metadata["feature_columns"]).to_json(
        local_dir / "feature_columns.json", orient="values"
    )
    with open(local_dir / "model_type.txt", "w") as f:
        f.write(model_type)

    hw_model = model_registry.python.create_model(
        name=model_name,
        metrics=metrics,
        description=f"AQI forecaster ({model_type}) — RMSE {metrics['rmse']:.2f}",
    )
    hw_model.save(str(local_dir))

    shutil.rmtree(local_dir)

    print(f"Registered {model_name} as {model_type} (new version created).")


def main():
    print("=" * 70)
    print("PEARLS AQI PREDICTOR — DAILY TRAINING PIPELINE")
    print("=" * 70)

    project = hopsworks.login()
    print("Project:", project.name)

    feature_store = project.get_feature_store()
    model_registry = project.get_model_registry()

    df = load_feature_store_data(feature_store)

    all_comparisons = []

    for horizon_label, horizon_hours in HORIZON_HOURS.items():
        print(f"\n{'#' * 70}")
        print(f"HORIZON: {horizon_label}")
        print(f"{'#' * 70}")

        dataset = build_supervised_dataset(df, horizon_hours)
        print(f"Supervised rows built: {len(dataset)}")

        train, test, cutoff_time = chronological_split(dataset)
        print(f"Train rows: {len(train)} | Test rows: {len(test)}")
        print(f"Chronological cutoff: {cutoff_time}")

        best_name, best_model, best_metrics, comparison_df = train_and_compare(
            train, test, horizon_label
        )

        comparison_df["horizon"] = horizon_label
        all_comparisons.append(comparison_df)

        register_model(
            model_registry,
            MODEL_NAMES[horizon_label],
            best_name,
            best_model,
            best_metrics,
            FEATURE_COLUMNS,
        )

    full_comparison = pd.concat(all_comparisons)
    output_path = PROJECT_ROOT / "model_comparison_report.csv"
    full_comparison.to_csv(output_path)

    print("\n" + "=" * 70)
    print("✅ DAILY TRAINING PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Comparison report saved to: {output_path}")


if __name__ == "__main__":
    main()
