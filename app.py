
import streamlit as st
import pandas as pd
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 0px;
}

.subtitle {
    font-size: 18px;
    color: #666666;
    margin-bottom: 25px;
}

.aqi-card {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #dddddd;
    background-color: #fafafa;
    text-align: center;
}

.aqi-number {
    font-size: 38px;
    font-weight: 700;
}

.aqi-category {
    font-size: 18px;
    font-weight: 600;
}

.section-title {
    font-size: 25px;
    font-weight: 650;
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# AQI FUNCTIONS
# ============================================================

def get_aqi_category(aqi):

    if aqi <= 50:
        return "Good"

    elif aqi <= 100:
        return "Moderate"

    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"

    elif aqi <= 200:
        return "Unhealthy"

    elif aqi <= 300:
        return "Very Unhealthy"

    else:
        return "Hazardous"


def get_aqi_emoji(category):

    mapping = {
        "Good": "🟢",
        "Moderate": "🟡",
        "Unhealthy for Sensitive Groups": "🟠",
        "Unhealthy": "🔴",
        "Very Unhealthy": "🟣",
        "Hazardous": "⚫"
    }

    return mapping.get(category, "⚪")


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).parent

FORECAST_FILE = PROJECT_DIR / "forecast_3days.csv"

SHAP_FILE = PROJECT_DIR / "xgboost_shap_summary.png"


# ============================================================
# LOAD FORECAST
# ============================================================

if not FORECAST_FILE.exists():

    st.error(
        "Forecast file not found: forecast_3days.csv"
    )

    st.stop()


forecast_df = pd.read_csv(FORECAST_FILE)

forecast_df["time"] = pd.to_datetime(
    forecast_df["time"],
    utc=True
)

forecast_df["aqi_category"] = (
    forecast_df["predicted_aqi"]
    .apply(get_aqi_category)
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🌍 Pearls AQI Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Based 3-Day Air Quality Forecasting'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Dashboard")

cities = sorted(
    forecast_df["city"].unique().tolist()
)

selected_city = st.sidebar.selectbox(
    "📍 Select City",
    cities
)

st.sidebar.divider()

st.sidebar.markdown("### 📌 Project")

st.sidebar.write(
    "Pearls AQI Predictor"
)

st.sidebar.write(
    "XGBoost-based AQI forecasting"
)

st.sidebar.write(
    "Forecast horizon: 3 days"
)


# ============================================================
# CITY DATA
# ============================================================

city_df = forecast_df[
    forecast_df["city"] == selected_city
].copy()

city_df = city_df.sort_values("time").reset_index(drop=True)


# ============================================================
# CURRENT FORECAST SUMMARY
# ============================================================

st.markdown(
    f'<div class="section-title">📍 {selected_city}</div>',
    unsafe_allow_html=True
)

first_row = city_df.iloc[0]

current_aqi = float(
    first_row["predicted_aqi"]
)

current_category = first_row[
    "aqi_category"
]

st.markdown(
    f"""
    <div class="aqi-card">
        <div>Next 24 Hours</div>
        <div class="aqi-number">{current_aqi:.1f}</div>
        <div class="aqi-category">
            {get_aqi_emoji(current_category)}
            {current_category}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown("### 📅 3-Day Forecast")


# ============================================================
# FORECAST CARDS
# ============================================================

cols = st.columns(3)

for i, (_, row) in enumerate(
    city_df.iterrows()
):

    with cols[i]:

        aqi = float(
            row["predicted_aqi"]
        )

        category = row[
            "aqi_category"
        ]

        date_text = row[
            "time"
        ].strftime("%d %b %Y")

        st.markdown(
            f"""
            <div class="aqi-card">

            <h3>{row["forecast"]}</h3>

            <div class="aqi-number">
                {aqi:.1f}
            </div>

            <div class="aqi-category">
                {get_aqi_emoji(category)}
                {category}
            </div>

            <p>{date_text}</p>

            </div>
            """,
            unsafe_allow_html=True
        )


st.divider()


# ============================================================
# AQI TREND
# ============================================================

st.markdown(
    '<div class="section-title">📈 AQI Forecast Trend</div>',
    unsafe_allow_html=True
)

chart_df = city_df[
    ["time", "predicted_aqi"]
].copy()

chart_df = chart_df.set_index(
    "time"
)

st.line_chart(
    chart_df,
    y="predicted_aqi",
    use_container_width=True
)


st.divider()


# ============================================================
# ALL CITY COMPARISON
# ============================================================

st.markdown(
    '<div class="section-title">🏙️ All Cities Comparison</div>',
    unsafe_allow_html=True
)

comparison_df = forecast_df.pivot(
    index="forecast",
    columns="city",
    values="predicted_aqi"
)

st.dataframe(
    comparison_df.round(2),
    use_container_width=True
)


# ============================================================
# AQI CATEGORY TABLE
# ============================================================

st.markdown("### 🚦 AQI Categories")

category_table = forecast_df[
    [
        "city",
        "forecast",
        "predicted_aqi",
        "aqi_category"
    ]
].copy()

category_table[
    "predicted_aqi"
] = category_table[
    "predicted_aqi"
].round(2)

st.dataframe(
    category_table,
    use_container_width=True,
    hide_index=True
)


st.divider()


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-title">🤖 Model Performance</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Model",
        "XGBoost"
    )

with col2:

    st.metric(
        "MAE",
        "1.8409"
    )

with col3:

    st.metric(
        "R² Score",
        "0.9934"
    )

st.caption(
    "Metrics are calculated on the historical test dataset."
)


# ============================================================
# MODEL COMPARISON
# ============================================================

st.markdown("### 🏆 Model Comparison")

model_comparison = pd.DataFrame({

    "Model": [
        "XGBoost",
        "Random Forest"
    ],

    "MAE": [
        1.8409,
        2.0517
    ],

    "RMSE": [
        3.0999,
        4.0670
    ],

    "R²": [
        0.9934,
        0.9886
    ]

})

st.dataframe(
    model_comparison,
    use_container_width=True,
    hide_index=True
)


st.divider()


# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

st.markdown(
    '<div class="section-title">🔍 XGBoost Explainability</div>',
    unsafe_allow_html=True
)

st.write(
    "SHAP analysis shows which features had the greatest "
    "impact on the XGBoost AQI predictions."
)

if SHAP_FILE.exists():

    st.image(
        str(SHAP_FILE),
        caption="XGBoost SHAP Feature Importance",
        use_container_width=True
    )

else:

    st.info(
        "SHAP summary plot is not available in the project folder yet."
    )


st.divider()


# ============================================================
# PROJECT SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">ℹ️ About the Project</div>',
    unsafe_allow_html=True
)

st.write(
    """
    **Pearls AQI Predictor** is an end-to-end machine learning
    project designed to forecast Air Quality Index (AQI) for
    major Pakistani cities.

    **Technology Stack**

    • Python  
    • Pandas / NumPy  
    • Scikit-learn  
    • XGBoost  
    • Random Forest  
    • SHAP  
    • Hopsworks Feature Store  
    • Streamlit  
    • Open-Meteo API  

    The final forecasting model is XGBoost because it achieved
    the best performance on the historical test dataset.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Pearls AQI Predictor • XGBoost • SHAP • Hopsworks • Streamlit"
)
