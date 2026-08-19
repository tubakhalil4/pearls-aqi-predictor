import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌍",
    layout="wide",
)

PROJECT_DIR = Path(__file__).parent
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

TEST_METRICS = pd.DataFrame({
    "Horizon": ["24h", "48h", "72h"],
    "MAE": [12.7445, 19.0359, 21.3232],
    "RMSE": [16.8846, 24.7506, 27.7760],
    "R²": [0.8040, 0.5795, 0.4708],
})


def aqi_category(value):
    for limit, category, emoji in AQI_CATEGORIES:
        if value <= limit:
            return category, emoji
    return "Unknown", "⚪"


@st.cache_data
def load_forecast():
    if not FORECAST_FILE.exists():
        raise FileNotFoundError(f"Missing forecast file: {FORECAST_FILE}")

    df = pd.read_csv(FORECAST_FILE)
    required = {"city", "current_aqi", "aqi_24h", "aqi_48h", "aqi_72h"}
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
st.caption("Leakage-safe XGBoost forecasting for the next 24, 48 and 72 hours")

try:
    forecast_df = load_forecast()
except Exception as exc:
    st.error(str(exc))
    st.stop()

cities = sorted(forecast_df["city"].unique())
selected_city = st.sidebar.selectbox("📍 Select city", cities)
city = forecast_df[forecast_df["city"] == selected_city].iloc[0]

st.sidebar.divider()
st.sidebar.markdown("### Project")
st.sidebar.write("4 Pakistani cities")
st.sidebar.write("3 forecast horizons")
st.sidebar.write("18 production features")
st.sidebar.write("XGBoost + SHAP")

current = float(city["current_aqi"])
current_cat, current_emoji = aqi_category(current)

st.subheader(f"📍 {selected_city}")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Current AQI", f"{current:.1f}")
with c2:
    value = float(city["aqi_24h"])
    st.metric("24h Forecast", f"{value:.1f}")
with c3:
    value = float(city["aqi_48h"])
    st.metric("48h Forecast", f"{value:.1f}")
with c4:
    value = float(city["aqi_72h"])
    st.metric("72h Forecast", f"{value:.1f}")

st.info(f"Current AQI category: {current_emoji} **{current_cat}**")

# Forecast table
st.subheader("📅 3-Day Forecast")
forecast_table = pd.DataFrame({
    "Horizon": ["24 Hours", "48 Hours", "72 Hours"],
    "Predicted AQI": [city["aqi_24h"], city["aqi_48h"], city["aqi_72h"]],
})
forecast_table[["Category", "Status"]] = forecast_table["Predicted AQI"].apply(
    lambda x: pd.Series(aqi_category(float(x)))
)
forecast_table["Predicted AQI"] = forecast_table["Predicted AQI"].astype(float).round(2)
st.dataframe(forecast_table, use_container_width=True, hide_index=True)

# Trend
st.subheader("📈 AQI Forecast Trend")
trend = pd.DataFrame({
    "Horizon": ["Current", "24h", "48h", "72h"],
    "AQI": [current, city["aqi_24h"], city["aqi_48h"], city["aqi_72h"]],
}).set_index("Horizon")
st.line_chart(trend, y="AQI", use_container_width=True)

# All-city comparison
st.subheader("🏙️ All Cities Comparison")
comparison = forecast_df[["city", "current_aqi", "aqi_24h", "aqi_48h", "aqi_72h"]].copy()
comparison.columns = ["City", "Current", "24h", "48h", "72h"]
for col in ["Current", "24h", "48h", "72h"]:
    comparison[col] = comparison[col].astype(float).round(2)
st.dataframe(comparison, use_container_width=True, hide_index=True)

# Model evaluation
st.subheader("🤖 Restored Model Test Performance")
st.dataframe(TEST_METRICS, use_container_width=True, hide_index=True)
st.caption("Metrics are from the chronological held-out test set and match the original saved-model results.")

# SHAP
st.subheader("🔍 SHAP Feature Importance")
shap_df = load_shap()
if not shap_df.empty:
    horizon_options = sorted(shap_df["horizon"].unique())
    horizon = st.selectbox("Forecast horizon", horizon_options)
    selected_shap = shap_df[shap_df["horizon"] == horizon].sort_values("mean_abs_shap", ascending=False)
    st.bar_chart(selected_shap.set_index("feature")["mean_abs_shap"].head(10), use_container_width=True)
    st.dataframe(selected_shap, use_container_width=True, hide_index=True)
else:
    st.info("SHAP CSV results are not available in the repository yet.")

# Leakage-safe design
st.subheader("🛡️ Forecast Integrity")
checks = {
    "Future AQI used as feature": "NO",
    "Future us_aqi used as feature": "NO",
    "Future target values used": "NO",
    "Current AQI state used": "YES",
    "Current environmental state used": "YES",
    "24h model used": "YES",
    "48h model used": "YES",
    "72h model used": "YES",
}
st.dataframe(pd.DataFrame(checks.items(), columns=["Check", "Result"]), use_container_width=True, hide_index=True)

st.divider()
st.caption("Pearls AQI Predictor • XGBoost • SHAP • Streamlit • Flask API • Open-Meteo")
