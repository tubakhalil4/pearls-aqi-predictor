# 🌍 Pearls AQI Predictor

An end-to-end machine learning system for forecasting the Air Quality Index (AQI) for Islamabad, Karachi, Lahore, and Peshawar for the next three days.

The project combines environmental data collection, feature engineering, leakage-safe multi-horizon forecasting, XGBoost models, SHAP explainability, a Streamlit dashboard, and a Flask API.

## Project Status

The core forecasting pipeline is complete and the final production forecast has been generated using the restored 24h, 48h, and 72h XGBoost models.

**Forecast origin:** `2026-08-19 05:00 UTC`

**Production features:** 18

The final pipeline explicitly avoids future-AQI leakage: future `us_aqi`/AQI values are not used as model inputs. Current August 2026 AQI-derived state and current environmental conditions are used instead.

## Supported Cities

- Islamabad
- Karachi
- Lahore
- Peshawar

## Final Forecast

| City | Current AQI | 24h | 48h | 72h |
|---|---:|---:|---:|---:|
| Islamabad | 132.00 | 128.25 | 122.82 | 118.57 |
| Karachi | 63.00 | 66.89 | 70.87 | 78.41 |
| Lahore | 143.00 | 132.18 | 130.19 | 126.85 |
| Peshawar | 140.00 | 130.69 | 122.40 | 121.96 |

## Model Validation

The saved XGBoost models were restored and validated against the original chronological test-set results.

| Horizon | MAE | RMSE | R² |
|---|---:|---:|---:|
| 24h | 12.7445 | 16.8846 | 0.8040 |
| 48h | 19.0359 | 24.7506 | 0.5795 |
| 72h | 21.3232 | 27.7760 | 0.4708 |

These are the final restored-model test metrics and should be used in project reporting. They replace earlier dashboard metrics that came from a different experiment.

## Leakage-Safe Forecasting

The final production forecast was verified with the following checks:

- Future AQI used as a feature: **No**
- Future `us_aqi` used as a feature: **No**
- Future target values used: **No**
- Current AQI state used: **Yes**
- Current environmental state used: **Yes**
- 24h XGBoost model used: **Yes**
- 48h XGBoost model used: **Yes**
- 72h XGBoost model used: **Yes**

## Production Features

The exact 18-feature schema is:

1. `pm2_5`
2. `pm10`
3. `carbon_monoxide`
4. `nitrogen_dioxide`
5. `ozone`
6. `sulphur_dioxide`
7. `temperature`
8. `humidity`
9. `wind_speed`
10. `wind_direction`
11. `pressure`
12. `rain`
13. `hour`
14. `day`
15. `month`
16. `day_of_week`
17. `aqi_change_rate`
18. `aqi_rolling_6h`

## Data Sources

The project uses Open-Meteo weather and air-quality APIs.

Environmental inputs include:

- PM2.5
- PM10
- Carbon monoxide
- Nitrogen dioxide
- Ozone
- Sulphur dioxide
- Temperature
- Relative humidity
- Precipitation/rain
- Wind speed
- Wind direction
- Atmospheric pressure

Open-Meteo's pre-computed `us_aqi` is used as the historical AQI target. AQI is not manually calculated in the pipeline.

## Feature Engineering

The pipeline creates:

- Time features: `hour`, `day`, `month`, `day_of_week`
- AQI change rate: `aqi_change_rate`
- Six-hour AQI rolling feature: `aqi_rolling_6h`

## SHAP Explainability

SHAP is used to explain the restored XGBoost forecasts. The analysis was completed separately for the 24h, 48h, and 72h models.

The strongest recurring feature was `aqi_rolling_6h`, followed by pollutant and time/environmental variables depending on the forecast horizon.

## Dashboard

The Streamlit dashboard provides:

- City selection
- Current AQI
- 24h, 48h, and 72h forecasts
- AQI category classification
- Forecast trend visualization
- All-city comparison
- Restored model test metrics
- SHAP feature importance
- Forecast-integrity/leakage checks

Run locally with:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Flask API

The repository also includes a Flask API.

Run it with:

```bash
python api.py
```

Endpoints:

```text
GET /
GET /api/forecast
GET /api/forecast/<city>
```

## AQI Categories

| AQI Range | Category |
|---:|---|
| 0–50 | Good |
| 51–100 | Moderate |
| 101–150 | Unhealthy for Sensitive Groups |
| 151–200 | Unhealthy |
| 201–300 | Very Unhealthy |
| 301+ | Hazardous |

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application and ML pipeline |
| Pandas / NumPy | Data processing |
| Scikit-learn | Model evaluation and ML utilities |
| XGBoost | Final AQI forecasting models |
| TensorFlow | Deep-learning experimentation |
| SHAP | Explainable AI |
| Hopsworks | Feature Store / ML infrastructure |
| Streamlit | Interactive dashboard |
| Flask | Forecast API |
| Open-Meteo | Environmental data |
| GitHub Actions | Repository validation / CI |

## Repository Structure

```text
pearls-aqi-predictor/
├── app.py
├── api.py
├── forecast_3days.csv
├── requirements.txt
├── runtime.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── shap/
│   └── shap_feature_importance_all_horizons.csv
└── .github/
    └── workflows/
        └── validate.yml
```

## Final Project Goal

Pearls AQI Predictor demonstrates an end-to-end AQI forecasting workflow: environmental data → feature engineering → historical model training/evaluation → saved multi-horizon XGBoost models → leakage-safe future inputs → 24h/48h/72h forecasts → SHAP explainability → Streamlit dashboard and Flask API.
