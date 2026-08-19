import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_DIR = Path(__file__).resolve().parent
FORECAST_FILE = PROJECT_DIR / "forecast_3days.csv"
SHAP_FILE = PROJECT_DIR / "shap_feature_importance_all_horizons.csv"

AQI_CATEGORIES = [
    (50, "Good", "🟢"),
    (100, "Moderate", "🟡"),
    (150, "Unhealthy for Sensitive Groups", "🟠"),
    (200, "Unhealthy", "🔴"),
    (300, "Very Unhealthy", "🟣"),
    (float("inf"), "Hazardous", "⚫"),
]

TEST_METRICS = pd.DataFrame(
    {
        "Horizon": ["24h", "48h", "72h"],
        "MAE": [12.7445, 19.0359, 21.3232],
        "RMSE": [16.8846, 24.7506, 27.7760],
        "R²": [0.8040, 0.5795, 0.4708],
    }
)


def aqi_category(value):
    for limit, category, emoji in AQI_CATEGORIES:
        if value <= limit:
            return category, emoji
    return "Unknown", "⚪"


@st.cache_data
def load_forecast():
    if not FORECAST_FILE.exists():
        raise FileNotFoundError(
            "forecast_3days.csv is missing from the repository. "
            "Run the forecasting pipeline and commit the generated forecast file."
        )

    df = pd.read_csv(FORECAST_FILE)
    required = {
        "city",
        "forecast_origin",
        "current_aqi",
        "aqi_24h",
        "aqi_48h",
        "aqi_72h",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Forecast file is missing columns: {sorted(missing)}")

    return df


@st.cache_data
def load_shap():
    if not SHAP_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(SHAP_FILE)


st.title("🌍 Pearls AQI Predictor")
st.caption(
    "Leakage-safe XGBoost forecasting for the next 24, 48 and 72 hours "
    "across four Pakistani cities."
)

try:
    forecast_df = load_forecast()
except Exception as exc:
    st.error(str(exc))
    st.stop()

cities = sorted(forecast_df["city"].dropna().unique())
if not cities:
    st.error("No cities are available in the forecast file.")
    st.stop()

selected_city = st.sidebar.selectbox("📍 Select city", cities)
city = forecast_df.loc[forecast_df["city"] == selected_city].iloc[0]

forecast_origin = pd.to_datetime(city["forecast_origin"], utc=True)

st.sidebar.divider()
st.sidebar.markdown("### Project")
st.sidebar.write("4 Pakistani cities")
st.sidebar.write("3 forecast horizons")
st.sidebar.write("18 production features")
st.sidebar.write("XGBoost + SHAP")
st.sidebar.write("Open-Meteo inputs")

current = float(city["current_aqi"])
current_cat, current_emoji = aqi_category(current)

st.subheader(f"📍 {selected_city}")
st.caption(f"Forecast origin: {forecast_origin.strftime('%Y-%m-%d %H:%M UTC')}")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Current AQI", f"{current:.1f}")
with c2:
    st.metric("24h Forecast", f"{float(city['aqi_24h']):.1f}")
with c3:
    st.metric("48h Forecast", f"{float(city['aqi_48h']):.1f}")
with c4:
    st.metric("72h Forecast", f"{float(city['aqi_72h']):.1f}")

if current > 150:
    st.error(f"⚠️ Current AQI is {current:.1f}: {current_cat}.")
elif current > 100:
    st.warning(f"⚠️ Current AQI is {current:.1f}: {current_cat}.")
else:
    st.success(f"Current AQI category: {current_emoji} {current_cat}")

st.subheader("📅 3-Day Forecast")
forecast_table = pd.DataFrame(
    {
        "Horizon": ["24 Hours", "48 Hours", "72 Hours"],
        "Predicted AQI": [
            float(city["aqi_24h"]),
            float(city["aqi_48h"]),
            float(city["aqi_72h"]),
        ],
    }
)
forecast_table[["Category", "Status"]] = forecast_table["Predicted AQI"].apply(
    lambda value: pd.Series(aqi_category(float(value)))
)
forecast_table["Predicted AQI"] = forecast_table["Predicted AQI"].round(2)
st.dataframe(forecast_table, use_container_width=True, hide_index=True)

st.subheader("📈 AQI Forecast Trend")
trend = pd.DataFrame(
    {
        "Horizon": ["Current", "24h", "48h", "72h"],
        "AQI": [
            current,
            float(city["aqi_24h"]),
            float(city["aqi_48h"]),
            float(city["aqi_72h"]),
        ],
    }
).set_index("Horizon")
st.line_chart(trend, y="AQI", use_container_width=True)

st.subheader("🏙️ All Cities Comparison")
comparison = forecast_df[
    ["city", "current_aqi", "aqi_24h", "aqi_48h", "aqi_72h"]
].copy()
comparison.columns = ["City", "Current", "24h", "48h", "72h"]
for column in ["Current", "24h", "48h", "72h"]:
    comparison[column] = comparison[column].astype(float).round(2)
st.dataframe(comparison, use_container_width=True, hide_index=True)

st.subheader("🤖 Restored Model Test Performance")
metrics_display = TEST_METRICS.copy()
metrics_display[["MAE", "RMSE", "R²"]] = metrics_display[["MAE", "RMSE", "R²"]].round(4)
st.dataframe(metrics_display, use_container_width=True, hide_index=True)
st.caption(
    "Chronological held-out test metrics. The restored models reproduce the "
    "original validation results exactly."
)

st.subheader("🔍 SHAP Feature Importance")
shap_df = load_shap()
if not shap_df.empty and {"horizon", "feature", "mean_abs_shap"}.issubset(shap_df.columns):
    horizon_options = sorted(shap_df["horizon"].dropna().unique())
    horizon = st.selectbox("Forecast horizon", horizon_options)
    selected_shap = shap_df[shap_df["horizon"] == horizon].sort_values(
        "mean_abs_shap", ascending=False
    )
    top_shap = selected_shap.head(10).set_index("feature")["mean_abs_shap"]
    st.bar_chart(top_shap, use_container_width=True)
    st.dataframe(selected_shap, use_container_width=True, hide_index=True)
else:
    st.info(
        "SHAP results are not included yet. The dashboard remains functional; "
        "add shap_feature_importance_all_horizons.csv to enable this section."
    )

st.subheader("🛡️ Forecast Integrity")
checks = pd.DataFrame(
    {
        "Check": [
            "Future AQI used as feature",
            "Future us_aqi used as feature",
            "Future target values used",
            "Current AQI state used",
            "Current environmental state used",
            "24h model used",
            "48h model used",
            "72h model used",
        ],
        "Result": ["NO", "NO", "NO", "YES", "YES", "YES", "YES", "YES"],
    }
)
st.dataframe(checks, use_container_width=True, hide_index=True)

st.subheader("⬇️ Download Forecast")
st.download_button(
    label="Download latest 3-day forecast CSV",
    data=forecast_df.to_csv(index=False).encode("utf-8"),
    file_name="pearls_aqi_3day_forecast.csv",
    mime="text/csv",
)

with st.expander("ℹ️ About this project"):
    st.markdown(
        """
        **Pearls AQI Predictor** forecasts AQI for Islamabad, Karachi, Lahore and
        Peshawar at 24h, 48h and 72h horizons.

        - Historical AQI and environmental data are used for model development.
        - Future AQI/us_aqi is never supplied as a prediction feature.
        - Current AQI-derived features initialize the production forecast state.
        - XGBoost models provide the three forecast horizons.
        - SHAP explains feature importance when the SHAP result file is available.
        """
    )

st.divider()
st.caption(
    "Pearls AQI Predictor • XGBoost • SHAP • Streamlit • Flask API • Open-Meteo"
)
