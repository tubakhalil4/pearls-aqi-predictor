#  Pearls AQI Predictor

**Pearls AQI Predictor** is an end-to-end machine-learning project that forecasts the Air Quality Index (AQI) for **Islamabad, Karachi, Lahore, and Peshawar** over the next **24, 48, and 72 hours**.

The final implementation combines environmental-data collection, feature engineering, chronological model validation, leakage-safe forecasting, XGBoost, SHAP explainability, a Streamlit dashboard, and a Flask API.

> **Final project state:** The forecasting pipeline, model validation, SHAP analysis, dashboard package, and Flask API are complete. Render deployment was not required for the final submission.

---

##  Project Objective

The project follows the provided Pearls AQI Predictor specification:

- collect historical and future environmental data
- build production-ready features
- train and evaluate AQI forecasting models
- forecast AQI at 24h, 48h, and 72h horizons
- prevent future-AQI leakage
- explain model predictions with SHAP
- expose results through a Streamlit dashboard and Flask API
- maintain the project in GitHub for reproducible submission

The final forecasting system uses **Open-Meteo** for environmental and air-quality data. Open-Meteo's pre-computed **US AQI (`us_aqi`)** is used as the AQI target; AQI is not manually calculated in the pipeline.

---

##  Supported Cities

- Islamabad
- Karachi
- Lahore
- Peshawar

---

##  Final Forecasting Pipeline

```text
Open-Meteo environmental data
          ↓
Historical / current AQI state
          ↓
Feature engineering
          ↓
18-feature production schema
          ↓
Restored XGBoost models
          ↓
Leakage-safe 24h / 48h / 72h forecasts
          ↓
SHAP explainability
          ↓
Streamlit Dashboard + Flask API
```

### Leakage-safe design

The production forecast was explicitly checked to ensure:

-  Future AQI is not used as a model feature
-  Future `us_aqi` is not used as a model feature
-  Future target values are not used
-  Current AQI state is used to initialize AQI-derived features
-  Current environmental conditions are used
-  Future weather/pollutant inputs are used without their future AQI target

This is important because the objective is genuine future forecasting rather than predicting AQI using information that would only be known after the forecast time.

---

##  Final Production Forecast

**Forecast origin:** `2026-08-19 05:00 UTC`

| City | Current AQI | 24h | 48h | 72h |
|---|---:|---:|---:|---:|
| Islamabad | 132.00 | 128.25 | 122.82 | 118.57 |
| Karachi | 63.00 | 66.89 | 70.87 | 78.41 |
| Lahore | 143.00 | 132.18 | 130.19 | 126.85 |
| Peshawar | 140.00 | 130.69 | 122.40 | 121.96 |

The forecast is stored in `forecast_3days.csv`.

---

##  Final Model Validation

The saved/restored XGBoost models were validated on a **chronological hold-out test set**. The restored models reproduced the original test metrics exactly.

| Forecast Horizon | MAE | RMSE | R² |
|---|---:|---:|---:|
| 24h | **12.7445** | **16.8846** | **0.8040** |
| 48h | **19.0359** | **24.7506** | **0.5795** |
| 72h | **21.3232** | **27.7760** | **0.4708** |

### Validation split

- Training rows: `56,140`
- Test rows: `14,036`
- Split type: chronological
- Training ended: `2025-08-07 18:00 UTC`
- Testing started: `2025-08-07 19:00 UTC`

The results show the expected reduction in predictive performance as the forecasting horizon increases, which is typical for longer-horizon AQI forecasting.

---

##  Production Feature Schema

The final production models use exactly **18 features**, in this order:

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

### Feature groups

**Pollutants**

- PM2.5
- PM10
- carbon monoxide
- nitrogen dioxide
- ozone
- sulphur dioxide

**Weather**

- temperature
- humidity
- wind speed
- wind direction
- pressure
- rain

**Time**

- hour
- day
- month
- day of week

**AQI-derived**

- `aqi_change_rate`
- `aqi_rolling_6h`

---

##  Data Source

The final pipeline uses the **Open-Meteo Weather API and Air Quality API**.

Environmental inputs include:

- PM2.5
- PM10
- carbon monoxide
- nitrogen dioxide
- ozone
- sulphur dioxide
- temperature
- relative humidity
- precipitation/rain
- wind speed
- wind direction
- atmospheric pressure

Open-Meteo's pre-computed `us_aqi` is used as the historical AQI target.

The original project specification mentioned AQICN or OpenWeather as possible external data sources. The implemented pipeline uses **Open-Meteo instead**, consistently for both historical and forecast environmental inputs.

---

##  Feature Engineering

The pipeline generates:

- `hour`
- `day`
- `month`
- `day_of_week`
- `aqi_change_rate`
- `aqi_rolling_6h`

For production forecasting, the AQI-derived features are initialized from the **latest available current AQI state**, rather than from an outdated historical observation.

The final August 2026 forecast used current AQI observations from the forecast origin of `2026-08-19 05:00 UTC`.

---

##  SHAP Explainability

SHAP was used to explain the restored XGBoost models separately for each forecast horizon.

The strongest recurring feature was:

- `aqi_rolling_6h`

Other important features varied by horizon and included pollutant, weather, and calendar variables such as PM2.5, ozone, carbon monoxide, sulphur dioxide, month, day, hour, and AQI change rate.

The combined SHAP results are stored at:

```text
shap/shap_feature_importance_all_horizons.csv
```

A visualization is also included:

```text
xgboost_shap_summary.png
```

---

##  Streamlit Dashboard

The repository contains a Streamlit dashboard that presents the final forecasting results.

Dashboard functionality includes:

- city selection
- current AQI
- 24h / 48h / 72h forecasts
- AQI category classification
- forecast comparison and visualization
- all-city comparison
- final model metrics
- SHAP feature importance
- forecast-integrity/leakage information

### Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

##  Flask API

A Flask API is included to expose the forecast data programmatically.

Run locally with:

```bash
python api.py
```

Endpoints:

```text
GET /
GET /api/forecast
GET /api/forecast/<city>
```

The API serves the generated forecast artifact from `forecast_3days.csv`.

---

##  AQI Categories

The dashboard uses the following US-AQI-style categories:

| AQI Range | Category |
|---:|---|
| 0–50 | Good |
| 51–100 | Moderate |
| 101–150 | Unhealthy for Sensitive Groups |
| 151–200 | Unhealthy |
| 201–300 | Very Unhealthy |
| 301+ | Hazardous |

---

##  Technology Stack

| Technology | Role in the project |
|---|---|
| Python | Application and ML pipeline |
| Pandas / NumPy | Data processing and feature engineering |
| Scikit-learn | Evaluation and ML utilities |
| XGBoost | Final 24h / 48h / 72h forecasting models |
| SHAP | Model explainability |
| Streamlit | Interactive dashboard |
| Flask | Forecast API |
| Open-Meteo | Environmental and air-quality data |
| Git / GitHub | Version control and project submission |
| GitHub Actions | Repository validation workflow |

TensorFlow was part of the broader technology specification/experimentation scope, but the **final production forecasting models documented here are XGBoost models**.

Hopsworks/Feature Store was part of the project's feature-store workflow, while the final repository submission focuses on the reproducible forecasting artifacts and application layer.

---

##  Repository Structure

```text
pearls-aqi-predictor/
├── app.py
├── api.py
├── forecast_3days.csv
├── requirements.txt
├── runtime.txt
├── Procfile
├── render.yaml
├── README.md
├── .gitignore
├── xgboost_shap_summary.png
├── shap/
│   └── shap_feature_importance_all_horizons.csv
├── .streamlit/
│   └── config.toml
└── .github/
    └── workflows/
        └── validate.yml
```

---

##  Installation

Clone the repository:

```bash
git clone https://github.com/tubakhalil4/pearls-aqi-predictor.git
cd pearls-aqi-predictor
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

Start the Flask API separately when required:

```bash
python api.py
```

---

##  Deployment Status

The repository includes deployment configuration files such as `Procfile`, `render.yaml`, and `runtime.txt`.

The application was validated locally/in the development environment, but **no public Render deployment is claimed as part of the final submission**.

This avoids representing an un-deployed service as a production URL.

---

##  Automation / CI-CD Scope

The repository contains a GitHub Actions validation workflow for repository-level checks.

The final submission should distinguish between:

- **implemented:** forecasting pipeline, validation, SHAP analysis, Streamlit dashboard, Flask API, repository validation
- **not claimed as live production infrastructure:** continuously scheduled hourly feature ingestion, daily automated retraining, and public Render hosting

These infrastructure extensions can be added later without changing the validated forecasting results.

---

##  Final Project Verification

The final repository package was verified with:

- required application files present
- `app.py` syntax valid
- `api.py` syntax valid
- dashboard dependencies available
- forecast CSV readable
- SHAP CSV readable
- four supported cities present
- exact 18-feature production schema preserved
- future AQI leakage checks passed
- 24h / 48h / 72h models available
- Git working tree clean

---

##  Final Project Outcome

Pearls AQI Predictor demonstrates a complete AQI forecasting workflow:

```text
Environmental data
        ↓
Feature engineering
        ↓
Historical model training & chronological evaluation
        ↓
Restored multi-horizon XGBoost models
        ↓
Current-state initialization
        ↓
Leakage-safe future environmental inputs
        ↓
24h / 48h / 72h AQI forecasts
        ↓
SHAP explainability
        ↓
Streamlit dashboard + Flask API
```

The final implementation is designed to provide an understandable, reproducible, and leakage-safe three-day AQI forecasting system for the four supported Pakistani cities.
