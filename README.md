# Pearls AQI Predictor

An end-to-end machine-learning project for forecasting the **Air Quality Index (AQI)** for four Pakistani cities:

* Islamabad
* Karachi
* Lahore
* Peshawar

The system produces AQI forecasts for:

* **Current AQI**
* **24-hour forecast**
* **48-hour forecast**
* **72-hour forecast**

The project combines historical environmental data collection, feature engineering, Hopsworks Feature Store, model training and comparison, model registration, hourly forecasting, SHAP explainability, a Streamlit dashboard, a Flask API, and GitHub Actions workflows.

---

## Project Overview

**Pearls AQI Predictor** is designed to demonstrate an end-to-end AQI forecasting workflow.

The implemented pipeline is:

```text
Open-Meteo Historical Data
        ↓
Historical Data Backfill
        ↓
Data Validation
        ↓
Feature Engineering
        ↓
Hopsworks Feature Store
        ↓
Supervised Forecast Dataset
        ↓
Model Training & Comparison
        ↓
Ridge / Random Forest / XGBoost
        ↓
Best Model Selected by RMSE
        ↓
Hopsworks Model Registry
        ↓
Hourly Production Forecast
        ↓
forecast_3days.csv
        ↓
Streamlit Dashboard + Flask API
```

The project uses **Open-Meteo's pre-computed `us_aqi` value as the AQI target**. AQI is therefore not manually calculated inside the forecasting pipeline.

---

## Project Objective

The main objective is to build a reproducible machine-learning workflow that can forecast AQI for the four supported cities at three future horizons.

The project implements:

* Historical environmental and air-quality data collection
* Data validation
* Feature engineering
* Hopsworks Feature Store integration
* 24-hour AQI forecasting
* 48-hour AQI forecasting
* 72-hour AQI forecasting
* Chronological train/test validation
* Comparison of multiple regression models
* Model registration in Hopsworks Model Registry
* Hourly production forecasting
* SHAP-based model explainability artifacts
* Streamlit dashboard
* Flask API
* GitHub Actions automation and validation

---

## Supported Cities

The forecasting system currently supports four cities:

| City      |
| --------- |
| Islamabad |
| Karachi   |
| Lahore    |
| Peshawar  |

The city is included as part of the data pipeline, while the trained forecasting features are numeric environmental, temporal, and AQI-state features.

---

# Data Sources

The project uses the **Open-Meteo Weather API** and **Open-Meteo Air Quality API**.

### Historical data

The historical backfill pipeline requests data from:

```text
2024-01-01
        ↓
2025-12-31
```

for all four supported cities.

Historical weather/environmental variables include:

* PM2.5
* PM10
* Carbon monoxide
* Nitrogen dioxide
* Ozone
* Sulphur dioxide
* Temperature
* Relative humidity
* Precipitation
* Wind speed
* Wind direction
* Atmospheric pressure

The air-quality API also provides:

```text
us_aqi
```

which is used as the historical AQI target.

### AQI target

The project does **not** manually calculate the AQI.

Instead:

```text
Open-Meteo us_aqi
        ↓
Historical AQI target
        ↓
Machine-learning target
```

This keeps the AQI definition consistent with the selected Open-Meteo data source.

---

# Historical Data Pipeline

The historical data collection pipeline is implemented in:

```text
pipelines/historical_backfill.py
```

It retrieves weather and air-quality data for:

* Islamabad
* Karachi
* Lahore
* Peshawar

The weather and air-quality responses are aligned using the hourly timestamp.

The resulting dataset is saved as:

```text
data/historical_raw_backfill.csv
```

The pipeline also prints:

* Dataset shape
* Rows per city
* Overall time range
* Missing-value counts
* Output location

---

# Feature Engineering

The production feature group is:

```text
aqi_features_v2
```

The project uses Hopsworks Feature Store version:

```text
Version: 1
```

The production forecasting schema contains **18 model features**.

## Production Features

### Pollutant features

```text
pm2_5
pm10
carbon_monoxide
nitrogen_dioxide
ozone
sulphur_dioxide
```

### Weather features

```text
temperature
humidity
wind_speed
wind_direction
pressure
rain
```

### Calendar features

```text
hour
day
month
day_of_week
```

### AQI-state features

```text
aqi_change_rate
aqi_rolling_6h
```

The complete feature order is:

```text
1.  pm2_5
2.  pm10
3.  carbon_monoxide
4.  nitrogen_dioxide
5.  ozone
6.  sulphur_dioxide
7.  temperature
8.  humidity
9.  wind_speed
10. wind_direction
11. pressure
12. rain
13. hour
14. day
15. month
16. day_of_week
17. aqi_change_rate
18. aqi_rolling_6h
```

The AQI itself is the target and is **not included as a direct model feature**.

---

# AQI-State Features

Two AQI-derived features are used:

### `aqi_change_rate`

Represents the recent change/trend in AQI.

### `aqi_rolling_6h`

Represents the recent six-hour AQI state.

These features are important because the current AQI state provides information about the recent trajectory of air quality.

At production inference time, these features are initialized from the currently available AQI state rather than from the future AQI target.

---

# Training Pipeline

The main training pipeline is:

```text
pipelines/training_pipeline.py
```

The pipeline connects to Hopsworks and reads:

```text
aqi_features_v2
version 1
```

The training process is repeated independently for:

```text
24h
48h
72h
```

---

# Supervised Forecast Dataset

For each forecasting horizon, the training pipeline creates a supervised dataset.

For an origin time:

```text
t0
```

and a forecast horizon:

```text
h
```

the target is:

```text
AQI(t0 + h)
```

The future environmental inputs are taken from the corresponding future timestamp.

The training formulation therefore attempts to reproduce the production forecasting setup:

```text
Origin time t0
       ↓
Current AQI state
       +
Future environmental/pollutant inputs
       +
Future calendar features
       ↓
Predicted AQI at t0 + h
```

The AQI target at the future timestamp is used as the **training target**, but not as an input feature.

---

# Leakage Consideration

Future-AQI leakage is an important issue in AQI forecasting.

A model would have invalid leakage if it used:

```text
Future AQI
```

or:

```text
Future us_aqi
```

as an input feature when predicting that same future AQI.

The project avoids directly using the future AQI target as a model feature.

Instead, the training formulation separates:

```text
Future environmental inputs
```

from:

```text
Future AQI target
```

The AQI-derived state features:

```text
aqi_change_rate
aqi_rolling_6h
```

are taken from the origin-time state.

Therefore, the intended structure is:

```text
Known at t0:
    current AQI state
    AQI trend features

Future input:
    future environmental/pollutant variables

Target:
    future AQI
```

This prevents direct future-AQI target leakage.

## Important limitation

The forecasting setup depends on having future environmental/pollutant inputs available.

During historical training, the future environmental values are available because the historical record already exists.

During real forecasting, those future environmental values must come from the production forecast/input source.

Therefore, model evaluation should be interpreted as evaluating the AQI prediction component **conditional on the future environmental inputs used by the forecasting pipeline**.

Errors in future environmental forecasts can also affect the final AQI prediction.

---

# Chronological Model Validation

The training pipeline does not randomly shuffle the time series before splitting.

Instead, it uses a chronological split:

```text
Earlier observations
        ↓
Training set

Later observations
        ↓
Test set
```

The default test fraction is:

```text
20%
```

This is more appropriate for time-dependent forecasting than a random train/test split because it evaluates the models on later observations.

The training pipeline records:

* Training rows
* Test rows
* Chronological cutoff
* Model metrics

---

# Models Compared

For every forecasting horizon, three regression models are trained.

## 1. Ridge Regression

```text
Ridge(alpha=1.0)
```

Ridge provides a regularized linear baseline.

## 2. Random Forest

The configured model uses:

```text
n_estimators = 300
max_depth = 12
random_state = 42
```

## 3. XGBoost

The configured XGBoost model uses:

```text
n_estimators = 400
max_depth = 6
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8
random_state = 42
```

---

# Model Selection

The three models are evaluated using:

* MAE
* RMSE
* R²

The model with the **lowest RMSE** is selected for each forecast horizon.

The selection process is:

```text
Ridge
   ↓
Random Forest
   ↓
XGBoost
   ↓
Compare RMSE
   ↓
Select lowest-RMSE model
```

The selected model is then registered in the Hopsworks Model Registry.

---

# Model Registry

The project creates separate registered models for each forecasting horizon.

```text
aqi_forecast_24h
aqi_forecast_48h
aqi_forecast_72h
```

Each registered model contains:

```text
model.pkl
feature_columns.json
model_type.txt
```

The model is serialized with `joblib`.

This allows the inference pipeline to restore the selected model using the same model format.

---

# Model Comparison Report

The training pipeline produces:

```text
model_comparison_report.csv
```

The report contains the model evaluation results for the supported horizons.

The main metrics are:

```text
RMSE
MAE
R²
horizon
```

This report is also committed by the daily training GitHub Actions workflow when the generated report changes.

---

# Recorded Model Validation Results

A recorded chronological hold-out evaluation produced the following results:

| Forecast Horizon |     MAE |    RMSE |     R² |
| ---------------- | ------: | ------: | -----: |
| 24h              | 12.7445 | 16.8846 | 0.8040 |
| 48h              | 19.0359 | 24.7506 | 0.5795 |
| 72h              | 21.3232 | 27.7760 | 0.4708 |

These values are a recorded validation snapshot, not a guarantee that every future retraining run will produce identical metrics.

The results show lower predictive performance as the forecast horizon increases.

---

# Exploratory Data Analysis

The EDA pipeline generates analysis and visualizations from the Hopsworks Feature Store data.

The EDA implementation analyzes:

* AQI trends over time
* Correlation between environmental variables and AQI
* Hourly AQI patterns
* Day-of-week AQI patterns
* Monthly/seasonal AQI patterns
* Per-city AQI summary statistics

EDA outputs are stored in:

```text
eda/
```

The generated visualizations include:

```text
01_aqi_trend_by_city.png
02_correlation_matrix.png
03_hourly_pattern.png
04_day_of_week_pattern.png
05_monthly_pattern.png
```

The generated report is:

```text
eda/EDA_REPORT.md
```

The EDA pipeline also calculates correlations between the available numeric environmental features and AQI.

---

# SHAP Explainability

SHAP explainability artifacts are included for the XGBoost forecasting models.

The combined feature-importance results are stored in:

```text
shap/shap_feature_importance_all_horizons.csv
```

A SHAP visualization is also included:

```text
xgboost_shap_summary.png
```

The explainability analysis helps identify which model features contribute most strongly to the XGBoost predictions.

The importance of features can vary between:

```text
24h
48h
72h
```

---

# Hourly Production Forecast

The production inference pipeline is:

```text
pipelines/hourly_forecast.py
```

It is designed to generate forecasts for the four supported cities.

The output contains:

```text
city
forecast_origin
current_aqi
aqi_24h
aqi_48h
aqi_72h
forecast_24h_time
forecast_48h_time
forecast_72h_time
```

The generated forecast file is:

```text
forecast_3days.csv
```

The output represents:

```text
Current AQI
      +
24-hour AQI forecast
      +
48-hour AQI forecast
      +
72-hour AQI forecast
```

for each supported city.

---

# Forecast Output

The forecast artifact contains one row per supported city.

The expected cities are:

```text
Islamabad
Karachi
Lahore
Peshawar
```

The forecast timestamps identify when each prediction applies.

The dashboard uses the generated forecast artifact to display the current AQI and future predictions.

---

# Streamlit Dashboard

The project includes a Streamlit dashboard in:

```text
app.py
```

The dashboard provides an interactive interface for viewing the generated AQI forecasts.

The implemented dashboard includes:

* City selection
* Current AQI
* 24-hour AQI forecast
* 48-hour AQI forecast
* 72-hour AQI forecast
* AQI category classification
* Forecast comparison
* All-city comparison
* Model metrics
* SHAP feature-importance information
* Forecast-integrity/leakage information

The deployed dashboard is available at:

https://pearls-aqipredictor.streamlit.app/

---

# Running the Dashboard

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit:

```bash
streamlit run app.py
```

---

# Flask API

The project also includes a Flask API:

```text
api.py
```

Run the API with:

```bash
python api.py
```

The API exposes forecast information from:

```text
forecast_3days.csv
```

Available endpoints include:

```text
GET /
GET /api/forecast
GET /api/forecast/<city>
```

---

# AQI Categories

The dashboard uses the following US-AQI-style categories:

|     AQI | Category                       |
| ------: | ------------------------------ |
|    0–50 | Good                           |
|  51–100 | Moderate                       |
| 101–150 | Unhealthy for Sensitive Groups |
| 151–200 | Unhealthy                      |
| 201–300 | Very Unhealthy                 |
|    301+ | Hazardous                      |

These categories are used for dashboard presentation.

---

# GitHub Actions Automation

The repository contains multiple GitHub Actions workflows.

## 1. Daily Model Training

Workflow:

```text
.github/workflows/daily_training.yml
```

The workflow is configured with:

```text
schedule:
    cron: "0 2 * * *"
```

and also supports:

```text
workflow_dispatch
```

The workflow:

1. Checks out the repository
2. Sets up Python
3. Installs project dependencies
4. Validates the training pipeline syntax
5. Runs the training pipeline
6. Validates `model_comparison_report.csv`
7. Commits the updated report when it changes

The training pipeline uses the Hopsworks credentials provided through:

```text
HOPSWORKS_API_KEY
```

as a GitHub Actions secret.

---

# Hourly Forecast Workflow

Workflow:

```text
.github/workflows/hourly_forecast.yml
```

The workflow is configured to run hourly:

```text
cron: "0 * * * *"
```

It also supports manual execution through:

```text
workflow_dispatch
```

The workflow:

1. Checks out the repository
2. Sets up Python
3. Installs dependencies
4. Validates `hourly_forecast.py`
5. Runs the production forecast
6. Validates `forecast_3days.csv`
7. Checks the four expected cities
8. Checks required forecast columns
9. Checks that AQI values are non-negative
10. Commits the forecast file when it changes

---

# Historical Backfill Workflow

Workflow:

```text
.github/workflows/historical_backfill.yml
```

This workflow is manually triggered with:

```text
workflow_dispatch
```

It runs the historical backfill and validation pipelines.

The workflow includes:

```text
historical_backfill.py
validate_historical_data.py
build_historical_features.py
validate_historical_features.py
```

This provides a separate workflow for preparing and validating historical production data.

---

# Repository Validation Workflow

Workflow:

```text
.github/workflows/validate.yml
```

The validation workflow runs on:

```text
push
pull_request
```

against the `main` branch.

It checks:

* Python syntax
* Forecast file availability
* Forecast schema
* Supported cities
* Forecast row count

This provides repository-level automated validation.

---

# Technology Stack

| Technology     | Use                                        |
| -------------- | ------------------------------------------ |
| Python         | Application and machine-learning pipelines |
| Pandas         | Data processing                            |
| NumPy          | Numerical processing                       |
| Scikit-learn   | Regression models and evaluation           |
| XGBoost        | Gradient-boosted regression                |
| Joblib         | Model serialization                        |
| Hopsworks      | Feature Store and Model Registry           |
| Open-Meteo     | Weather and air-quality data               |
| SHAP           | Model explainability                       |
| Streamlit      | Interactive dashboard                      |
| Flask          | Forecast API                               |
| Git            | Version control                            |
| GitHub         | Source-code repository                     |
| GitHub Actions | Automated workflows                        |

---

# Repository Structure

```text
pearls-aqi-predictor/
│
├── app.py
├── api.py
├── requirements.txt
├── README.md
├── forecast_3days.csv
├── model_comparison_report.csv
│
├── data/
│   └── historical_raw_backfill.csv
│
├── pipelines/
│   ├── historical_backfill.py
│   ├── validate_historical_data.py
│   ├── build_historical_features.py
│   ├── validate_historical_features.py
│   ├── training_pipeline.py
│   ├── hourly_forecast.py
│   └── ...
│
├── eda/
│   ├── 01_aqi_trend_by_city.png
│   ├── 02_correlation_matrix.png
│   ├── 03_hourly_pattern.png
│   ├── 04_day_of_week_pattern.png
│   ├── 05_monthly_pattern.png
│   └── EDA_REPORT.md
│
├── shap/
│   └── shap_feature_importance_all_horizons.csv
│
├── xgboost_shap_summary.png
│
├── .streamlit/
│   └── config.toml
│
└── .github/
    └── workflows/
        ├── daily_training.yml
        ├── hourly_forecast.yml
        ├── historical_backfill.yml
        └── validate.yml
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/tubakhalil4/pearls-aqi-predictor.git
cd pearls-aqi-predictor
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Training Pipeline

The training pipeline requires Hopsworks credentials.

The required environment variables are:

```text
HOPSWORKS_PROJECT
HOPSWORKS_API_KEY
```

Example:

```bash
export HOPSWORKS_PROJECT="Tuba_Pearl_Project"
export HOPSWORKS_API_KEY="YOUR_HOPSWORKS_API_KEY"
```

Then run:

```bash
python pipelines/training_pipeline.py
```

The pipeline will:

```text
Read Feature Store
      ↓
Build 24h dataset
      ↓
Train and compare models
      ↓
Register best 24h model
      ↓
Build 48h dataset
      ↓
Train and compare models
      ↓
Register best 48h model
      ↓
Build 72h dataset
      ↓
Train and compare models
      ↓
Register best 72h model
      ↓
Save model_comparison_report.csv
```

---

# Running the Production Forecast

Set the Hopsworks environment variables and run:

```bash
python pipelines/hourly_forecast.py
```

The expected output is:

```text
forecast_3days.csv
```

---

# Running EDA

The EDA pipeline requires access to the Hopsworks Feature Store.

Run:

```bash
python pipelines/eda.py
```

The exact EDA script name should match the pipeline file present in the repository.

Generated charts and the markdown report are stored in:

```text
eda/
```

---

# Running the Flask API

```bash
python api.py
```

The API reads:

```text
forecast_3days.csv
```

and exposes the forecast through HTTP endpoints.

---

# Project Limitations

The current implementation has several limitations.

## 1. Four cities only

The system currently supports:

```text
Islamabad
Karachi
Lahore
Peshawar
```

Additional cities would require changes to the data collection and forecasting configuration.

## 2. Forecast quality depends on future environmental inputs

The training formulation uses future environmental/pollutant variables as model inputs.

In real forecasting, those inputs must be available from the production environmental forecast source.

Therefore:

```text
Environmental forecast error
        ↓
AQI input error
        ↓
Potential AQI prediction error
```

## 3. Longer horizons are less accurate

The recorded validation results show lower R² and higher error as the forecast horizon increases.

The recorded results were:

```text
24h → RMSE 16.8846, R² 0.8040
48h → RMSE 24.7506, R² 0.5795
72h → RMSE 27.7760, R² 0.4708
```

Therefore the 72-hour forecast should not be interpreted as being as accurate as the 24-hour forecast.

## 4. External service dependency

The training and production pipelines depend on external services such as:

```text
Open-Meteo
Hopsworks
GitHub Actions
```

Failures or availability problems in these services can prevent the corresponding workflow from completing successfully.

## 5. Model performance is not guaranteed to remain constant

The model is retrained through the training workflow.

Because new data can change the training distribution, future training runs can produce different:

* model selections
* RMSE
* MAE
* R²
* forecasts

The recorded metrics in this README should therefore be treated as a validation snapshot.

---

# Leakage Limitations and Interpretation

The project specifically addresses direct future-AQI leakage.

The model does not use:

```text
future AQI
future us_aqi
```

as prediction features.

However, the training formulation uses future environmental/pollutant values because the intended production architecture obtains those values from future environmental forecasts.

This creates an important evaluation condition:

```text
AQI model performance
        +
quality of future environmental inputs
        ↓
final production forecast quality
```

Therefore, a complete future improvement would be to evaluate the entire forecasting chain using the actual historical environmental forecasts that would have been available at each historical forecast origin, rather than using realized future environmental observations.

---

# Possible Future Improvements

The current project can be improved in several areas.

## 1. Historical forecast simulation

Instead of training/evaluating with realized future environmental observations, a stronger evaluation framework would reconstruct what environmental forecasts were actually available at each historical forecast origin.

This would provide a more realistic end-to-end assessment.

## 2. More advanced time-series features

Additional lag and rolling features could be evaluated, such as:

```text
AQI lag 1h
AQI lag 3h
AQI lag 6h
AQI lag 12h
AQI lag 24h
Rolling 12h AQI
Rolling 24h AQI
```

These should only be added if they are constructed strictly from information available at the forecast origin.

## 3. Better temporal validation

Instead of a single chronological hold-out split, future work could use rolling or walk-forward validation.

Example:

```text
Train → Test
Train + Test → Next Test
Train + previous Test → Next Test
```

This would provide multiple historical evaluation periods.

## 4. Hyperparameter tuning

The current model parameters are explicitly configured.

Future work could evaluate systematic hyperparameter optimization for:

* Ridge
* Random Forest
* XGBoost

while preserving chronological validation.

## 5. Additional forecasting models

Future experiments could compare the current models against additional time-series or machine-learning approaches.

Any additional model should still be evaluated using the same leakage-safe temporal methodology.

## 6. Prediction uncertainty

The current output provides point forecasts.

A future version could provide prediction intervals so that the dashboard communicates forecast uncertainty.

## 7. More cities

The data collection configuration could be extended to additional cities if required.

## 8. Monitoring

Future versions could monitor:

* Forecast error over time
* Feature drift
* AQI distribution drift
* Missing input data
* Workflow failures
* Model performance changes

---

# Reproducibility

The project is organized so that the main stages can be reproduced through scripts and GitHub Actions.

The main reproducibility flow is:

```text
Historical Backfill
        ↓
Historical Validation
        ↓
Production Feature Construction
        ↓
Feature Validation
        ↓
Model Training
        ↓
Model Comparison
        ↓
Model Registration
        ↓
Production Forecast
        ↓
Forecast Validation
        ↓
Dashboard / API
```

The repository stores important generated artifacts such as:

```text
forecast_3days.csv
model_comparison_report.csv
SHAP outputs
EDA outputs
```

---

# Project Deliverables

The current project delivers:

### Data pipeline

```text
Historical Open-Meteo data collection
Historical validation
Production feature construction
Feature validation
```

### Machine learning

```text
24h AQI forecasting
48h AQI forecasting
72h AQI forecasting

Ridge comparison
Random Forest comparison
XGBoost comparison

RMSE / MAE / R² evaluation
```

### Model management

```text
Hopsworks Feature Store
Hopsworks Model Registry
Joblib model serialization
```

### Explainability

```text
SHAP feature-importance artifacts
```

### Applications

```text
Streamlit dashboard
Flask API
```

### Automation

```text
Daily training workflow
Hourly forecast workflow
Historical backfill workflow
Repository validation workflow
```

---

# Final System Architecture

The implemented architecture can be summarized as:

```text
                 OPEN-METEO
                     │
          ┌──────────┴──────────┐
          │                     │
    Historical Data       Forecast Inputs
          │                     │
          ↓                     ↓
 Historical Backfill      Hourly Forecast
          │                     │
          ↓                     │
 Feature Engineering           │
          │                     │
          ↓                     │
 Hopsworks Feature Store       │
          │                     │
          ↓                     │
 Training Pipeline             │
          │                     │
   ┌──────┼───────┐             │
   ↓      ↓       ↓             │
 Ridge    RF    XGBoost         │
   │      │       │             │
   └──────┼───────┘             │
          ↓                     │
   Lowest RMSE Model            │
          │                     │
          ↓                     │
 Hopsworks Model Registry       │
          │                     │
          └──────────┬──────────┘
                     ↓
             24h / 48h / 72h
               AQI Forecasts
                     │
                     ↓
             forecast_3days.csv
                 │          │
                 ↓          ↓
            Streamlit     Flask API
             Dashboard
```

---

# Conclusion

Pearls AQI Predictor implements an end-to-end AQI forecasting workflow for Islamabad, Karachi, Lahore, and Peshawar.

The system:

```text
Collects environmental data
        ↓
Builds production features
        ↓
Stores features in Hopsworks
        ↓
Creates multi-horizon supervised datasets
        ↓
Compares Ridge, Random Forest and XGBoost
        ↓
Selects the lowest-RMSE model
        ↓
Registers forecasting models
        ↓
Generates 24h / 48h / 72h AQI forecasts
        ↓
Produces SHAP explainability artifacts
        ↓
Validates forecast outputs
        ↓
Displays forecasts through Streamlit
        ↓
Exposes forecasts through Flask
        ↓
Automates training, forecasting and validation with GitHub Actions
```

The project explicitly addresses direct future-AQI leakage by keeping future AQI targets separate from model input features. At the same time, the forecasting formulation depends on future environmental/pollutant inputs, meaning the quality of those inputs is an important limitation of the complete forecasting system.

The recorded validation results demonstrate that the 24-hour horizon performs better than the 48-hour and 72-hour horizons on the recorded chronological hold-out evaluation.

The project therefore provides a complete working foundation for multi-horizon AQI forecasting while leaving clear opportunities for stronger end-to-end historical forecast simulation, temporal validation, uncertainty estimation, monitoring, and model improvement.
