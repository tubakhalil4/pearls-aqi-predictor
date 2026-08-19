from pathlib import Path

import pandas as pd
from flask import Flask, jsonify

app = Flask(__name__)

FORECAST_FILE = Path(__file__).parent / "forecast_3days.csv"


def load_forecast():
    if not FORECAST_FILE.exists():
        raise FileNotFoundError("forecast_3days.csv not found")
    return pd.read_csv(FORECAST_FILE)


@app.get("/")
def root():
    return jsonify({
        "project": "Pearls AQI Predictor",
        "status": "ok",
        "service": "Flask AQI Forecast API",
        "endpoints": ["/api/forecast", "/api/forecast/<city>"],
    })


@app.get("/api/forecast")
def all_forecasts():
    df = load_forecast()
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/forecast/<city>")
def city_forecast(city):
    df = load_forecast()
    result = df[df["city"].str.lower() == city.lower()]

    if result.empty:
        return jsonify({"error": f"City not found: {city}"}), 404

    return jsonify(result.iloc[0].to_dict())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
