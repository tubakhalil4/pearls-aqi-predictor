import os
from pathlib import Path

import hopsworks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EDA_DIR = PROJECT_ROOT / "eda"
EDA_DIR.mkdir(exist_ok=True)

FEATURE_GROUP_NAME = "aqi_features_v2"
FEATURE_GROUP_VERSION = 1

NUMERIC_FEATURES = [
    "pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide",
    "ozone", "sulphur_dioxide", "temperature", "humidity",
    "wind_speed", "wind_direction", "pressure", "rain",
]


def load_data(feature_store):
    fg = feature_store.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
    )
    df = fg.read()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values(["city", "time"]).reset_index(drop=True)
    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek
    df["month"] = df["time"].dt.month
    df["date"] = df["time"].dt.date
    return df


def plot_trend(df, out_path):
    daily = df.groupby(["date", "city"])["aqi"].mean().reset_index()
    daily["date"] = pd.to_datetime(daily["date"])

    fig, ax = plt.subplots(figsize=(12, 5))
    for city, g in daily.groupby("city"):
        ax.plot(g["date"], g["aqi"], label=city, linewidth=1)
    ax.set_title("Daily Average AQI by City (2024 - present)")
    ax.set_xlabel("Date")
    ax.set_ylabel("AQI")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_correlation(df, out_path):
    corr_cols = NUMERIC_FEATURES + ["aqi"]
    corr = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_cols)))
    ax.set_yticks(range(len(corr_cols)))
    ax.set_xticklabels(corr_cols, rotation=45, ha="right")
    ax.set_yticklabels(corr_cols)
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=7)
    ax.set_title("Correlation Matrix: Weather/Pollutants vs AQI")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    return corr["aqi"].drop("aqi").sort_values(key=abs, ascending=False)


def plot_hourly_pattern(df, out_path):
    hourly = df.groupby(["hour", "city"])["aqi"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    for city, g in hourly.groupby("city"):
        ax.plot(g["hour"], g["aqi"], marker="o", markersize=3, label=city)
    ax.set_title("Average AQI by Hour of Day")
    ax.set_xlabel("Hour (UTC)")
    ax.set_ylabel("Average AQI")
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    return hourly.groupby("hour")["aqi"].mean()


def plot_dow_pattern(df, out_path):
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow = df.groupby(["day_of_week", "city"])["aqi"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    for city, g in dow.groupby("city"):
        ax.plot(g["day_of_week"], g["aqi"], marker="o", markersize=4, label=city)
    ax.set_title("Average AQI by Day of Week")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Average AQI")
    ax.set_xticks(range(7))
    ax.set_xticklabels(dow_names)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    return dow.groupby("day_of_week")["aqi"].mean()


def plot_monthly_pattern(df, out_path):
    monthly = df.groupby(["month", "city"])["aqi"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    for city, g in monthly.groupby("city"):
        ax.plot(g["month"], g["aqi"], marker="o", markersize=4, label=city)
    ax.set_title("Average AQI by Month (Seasonal Pattern)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average AQI")
    ax.set_xticks(range(1, 13))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    return monthly.groupby("month")["aqi"].mean()


def city_summary_stats(df):
    return df.groupby("city")["aqi"].agg(["mean", "std", "min", "max"]).round(2)


def main():
    print("=" * 70)
    print("PEARLS AQI PREDICTOR — EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    project = hopsworks.login()
    feature_store = project.get_feature_store()

    df = load_data(feature_store)
    print(f"Rows loaded: {len(df)}")
    print(f"Cities: {sorted(df['city'].unique())}")
    print(f"Date range: {df['time'].min()} -> {df['time'].max()}")

    # ---- Trend ----
    plot_trend(df, EDA_DIR / "01_aqi_trend_by_city.png")
    print("Saved: 01_aqi_trend_by_city.png")

    # ---- Correlation ----
    aqi_corr = plot_correlation(df, EDA_DIR / "02_correlation_matrix.png")
    print("Saved: 02_correlation_matrix.png")
    print("\nCorrelation with AQI (sorted by strength):")
    print(aqi_corr.to_string())

    # ---- Hourly pattern ----
    hourly_means = plot_hourly_pattern(df, EDA_DIR / "03_hourly_pattern.png")
    print("Saved: 03_hourly_pattern.png")
    peak_hour = hourly_means.idxmax()
    low_hour = hourly_means.idxmin()

    # ---- Day-of-week pattern ----
    dow_means = plot_dow_pattern(df, EDA_DIR / "04_day_of_week_pattern.png")
    print("Saved: 04_day_of_week_pattern.png")

    # ---- Monthly / seasonal pattern ----
    monthly_means = plot_monthly_pattern(df, EDA_DIR / "05_monthly_pattern.png")
    print("Saved: 05_monthly_pattern.png")
    peak_month = monthly_means.idxmax()
    low_month = monthly_means.idxmin()

    # ---- City summary stats ----
    summary = city_summary_stats(df)
    print("\nPer-city AQI summary:")
    print(summary.to_string())

    # ---- Build markdown report with real computed numbers ----
    top_positive = aqi_corr[aqi_corr > 0].head(3)
    top_negative = aqi_corr[aqi_corr < 0].head(3)

    report_lines = []
    report_lines.append("# Exploratory Data Analysis — Pearls AQI Predictor\n")
    report_lines.append(
        f"Data source: Hopsworks Feature Store (`{FEATURE_GROUP_NAME}` v{FEATURE_GROUP_VERSION}), "
        f"{len(df)} rows, {df['time'].min().date()} to {df['time'].max().date()}, "
        f"covering {', '.join(sorted(df['city'].unique()))}.\n"
    )

    report_lines.append("## 1. AQI Trend Over Time\n")
    report_lines.append("![AQI Trend](01_aqi_trend_by_city.png)\n")
    report_lines.append("Per-city AQI summary statistics:\n")
    report_lines.append(summary.to_markdown() + "\n")

    report_lines.append("## 2. Correlation with AQI\n")
    report_lines.append("![Correlation Matrix](02_correlation_matrix.png)\n")
    report_lines.append(
        "Features most positively correlated with AQI: "
        + ", ".join(f"{k} (r={v:.2f})" for k, v in top_positive.items())
        + ".\n"
    )
    report_lines.append(
        "Features most negatively correlated with AQI: "
        + ", ".join(f"{k} (r={v:.2f})" for k, v in top_negative.items())
        + ".\n"
    )

    report_lines.append("## 3. Hourly Pattern\n")
    report_lines.append("![Hourly Pattern](03_hourly_pattern.png)\n")
    report_lines.append(
        f"Averaged across all cities, AQI peaks around hour {int(peak_hour)}:00 UTC "
        f"and is lowest around hour {int(low_hour)}:00 UTC.\n"
    )

    report_lines.append("## 4. Day-of-Week Pattern\n")
    report_lines.append("![Day of Week Pattern](04_day_of_week_pattern.png)\n")

    report_lines.append("## 5. Seasonal (Monthly) Pattern\n")
    report_lines.append("![Monthly Pattern](05_monthly_pattern.png)\n")
    month_names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    report_lines.append(
        f"AQI is highest on average in {month_names[int(peak_month)]} "
        f"and lowest in {month_names[int(low_month)]}, consistent with seasonal "
        f"pollution patterns (e.g. winter inversion / crop-burning season effects "
        f"common in Pakistani cities, if applicable to the observed peak month).\n"
    )

    report_path = EDA_DIR / "EDA_REPORT.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"\n✅ EDA complete. Report written to: {report_path}")
    print(f"Plots saved in: {EDA_DIR}")


if __name__ == "__main__":
    main()
