Pearls AQI Predictor

An end-to-end, automated machine-learning system that forecasts the Air Quality Index (AQI) for Islamabad, Karachi, Lahore, and Peshawar at 24-hour, 48-hour, and 72-hour horizons.

Live dashboard: https://pearls-aqipredictor.streamlit.app/
Full project documentation: see Pearls_AQI_Predictor_Documentation.pdf in this repository

## Overview

The system collects live weather and air-quality data from Open-Meteo, stores engineered features in a Hopsworks Feature Store, trains and compares Ridge Regression, Random Forest, and XGBoost models daily, and serves predictions through a Streamlit dashboard and Flask API. Both the feature/forecast pipeline and the training pipeline run automatically via GitHub Actions.

Open-Meteo API → Hourly Feature Pipeline (every hour) → Hopsworks Feature Store (aqi_features_v2) → Daily Training Pipeline (daily) and Hourly Forecast Pipeline → Hopsworks Model Registry and forecast_3days.csv → Streamlit Dashboard + Flask API

## Supported Cities

Islamabad, Karachi, Lahore, Peshawar

## Data Source

Open-Meteo Weather API and Air Quality API. Open-Meteo's pre-computed us_aqi is used directly as the AQI target.

## Production Feature Schema

18 features: 6 pollutants (PM2.5, PM10, CO, NO2, O3, SO2), 6 weather variables (temperature, humidity, wind speed, wind direction, pressure, rain), 4 calendar features (hour, day, month, day of week), and 2 AQI-state features (aqi_change_rate, aqi_rolling_6h).

## Model Comparison and Selection

Ridge Regression, Random Forest, and XGBoost are trained and compared daily on a chronologically held-out test split. The lowest-RMSE model per horizon is registered to production.

24h: XGBoost selected, RMSE 13.39, MAE 10.02, R2 0.877
48h: XGBoost selected, RMSE 16.06, MAE 12.36, R2 0.823
72h: XGBoost selected, RMSE 16.78, MAE 13.00, R2 0.806

Latest results always available in model_comparison_report.csv.

## Automation

Hourly Feature and Forecast Pipeline (.github/workflows/hourly_forecast.yml, every hour)
Daily Training Pipeline (.github/workflows/daily_training.yml, daily at 02:00 UTC)
Historical Backfill (.github/workflows/historical_backfill.yml, manual trigger)
Continuous Validation (.github/workflows/validate.yml, on every push)

## Exploratory Data Analysis

PM2.5 is the strongest correlate with AQI (r=0.75); wind speed is the strongest negative correlate (r=-0.32). Lahore is the most polluted and volatile city (mean AQI 151, max 537). Analysis script is at pipelines/eda_analysis.py, full report and charts are in eda/EDA_REPORT.md.

## SHAP Explainability

aqi_rolling_6h is the strongest recurring driver across all three horizons. Results in shap/shap_feature_importance_all_horizons.csv and xgboost_shap_summary.png.

## Dashboard

Streamlit dashboard (app.py) showing current AQI, 24h/48h/72h forecasts, hazard alerts, AQI categorization, all-city comparison, live model performance, SHAP importance, and forecast-integrity checks.

Run locally: pip install -r requirements.txt then streamlit run app.py

## Flask API

Run with: python api.py
Endpoints: GET /, GET /api/forecast, GET /api/forecast/<city>

## AQI Categories

0-50 Good, 51-100 Moderate, 101-150 Unhealthy for Sensitive Groups, 151-200 Unhealthy, 201-300 Very Unhealthy, 301+ Hazardous

## Technology Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, Joblib, Hopsworks, Open-Meteo, SHAP, Streamlit, Flask, GitHub Actions.

## Repository Structure

app.py, api.py, requirements.txt, runtime.txt, README.md, Pearls_AQI_Predictor_Documenation.pdf, forecast_3days.csv, model_comparison_report.csv, xgboost_shap_summary.png, pipelines folder containing historical_backfill.py, build_historical_features.py, validate_historical_data.py, validate_historical_features.py, hourly_forecast.py, training_pipeline.py, eda_analysis.py, eda folder containing EDA_REPORT.md and five PNG charts, shap folder containing shap_feature_importance_all_horizons.csv, .streamlit folder containing config.toml, and .github/workflows folder containing historical_backfill.yml, hourly_forecast.yml, daily_training.yml, validate.yml



## Known Limitations

Model performance depends on the accuracy of Open-Meteo's own future weather/pollutant forecasts, since these are used as model inputs. Longer horizons are less accurate than shorter ones, as shown in the comparison table above. Hopsworks' free-tier job queue can add a few minutes of latency per feature store write.

## Project Report

For full documentation including the data leakage investigation, model exploration (Ridge, Random Forest, XGBoost, and an LSTM deep learning experiment), detailed EDA, and complete limitations, see Pearls_AQI_Predictor_Documentation.pdf in this repository.
