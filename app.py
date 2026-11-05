from __future__ import annotations

from io import StringIO
from time import sleep
from typing import Callable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from core.data_loader import (
    load_global_temperature,
    load_alternative_temperature,
    load_gridded_temperature,
)
from core.forecasting import MODEL_NAMES, forecast_model, evaluate_models
from core.analytics import annual_summary, detect_records, regional_series
from core.explainability import explain_random_forest
from core.theme import (
    BLUE, GREY, RED,
    header, importance_fig, inject_theme, line_fig, plot, warming_stripes,
)


# ---------------------------------------------------------------------
# Page configuration and constants
# ---------------------------------------------------------------------

st.set_page_config(page_title="NASA Climate Outlook", page_icon="🌍", layout="wide")

MONTHS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

NASA_GISTEMP_URL = (
    "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
)
NASA_GISTEMP_MIRROR_URL = (
    "https://psl.noaa.gov/data/timeseries/month/data/nasa.land.csv"
)


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def get_global(source: str) -> pd.DataFrame:
    """Load the selected source, falling back to NOAA if NASA is unavailable."""
    if source == "NOAA PSL mirror":
        return load_alternative_temperature()

    try:
        return load_global_temperature()
    except Exception as primary_error:
        try:
            fallback = load_alternative_temperature()
            fallback.attrs["source"] = (
                "NOAA PSL mirror (automatic fallback after NASA source error)"
            )
            fallback.attrs["fallback_reason"] = str(primary_error)
            return fallback
        except Exception as fallback_error:
            raise RuntimeError(
                "NASA GISTEMP and the NOAA PSL fallback were unavailable. "
                f"NASA error: {primary_error}; NOAA error: {fallback_error}"
            ) from fallback_error


# ---------------------------------------------------------------------
# Original dashboard calculations — retained
# ---------------------------------------------------------------------

def seasonal_forecast(series: pd.Series, horizon: int) -> pd.Series:
    """Forecast each calendar month from its recent seasonal mean and trend."""
    clean = series.dropna().astype(float)
    if len(clean) < 36:
        raise ValueError("At least 36 monthly observations are required.")

    recent = clean.iloc[-min(len(clean), 180):]
    month_means = recent.groupby(recent.index.month).mean()
    available = month_means.reindex(np.arange(1, 13)).interpolate().bfill().ffill()

    trend_window = min(len(recent), 120)
    slope = float(
        np.polyfit(
            np.arange(trend_window),
            recent.iloc[-trend_window:].to_numpy(),
            1,
        )[0]
    )
    dates = pd.date_range(
        clean.index[-1] + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )
    values = [
        available.loc[date.month] + slope * step
        for step, date in enumerate(dates, start=1)
    ]
    return pd.Series(values, index=dates, name="forecast_c")


def persistence_forecast(series: pd.Series, horizon: int) -> pd.Series:
    clean = series.dropna().astype(float)
    if clean.empty:
        raise ValueError("At least one observation is required.")
    dates = pd.date_range(
        clean.index[-1] + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )
    return pd.Series(clean.iloc[-1], index=dates, name="forecast_c")


def annualized_slope(series: pd.Series, months: int) -> float:
    clean = series.dropna().astype(float).iloc[-months:]
    if len(clean) < 3:
        return float("nan")
    return float(np.polyfit(np.arange(len(clean)), clean.to_numpy(), 1)[0] * 12)


def rising_frequency(
    series: pd.Series,
    months: int = 24,
    window: int = 12,
) -> float:
    clean = series.dropna().astype(float)
    if len(clean) < months + window:
        return float("nan")
    slopes = []
    for end in range(months, len(clean) + 1):
        sample = clean.iloc[end - months:end]
        slopes.append(annualized_slope(sample, window))
    return float(np.mean(np.asarray(slopes) > 0.02))


def rolling_mae(
    series: pd.Series,
    forecast_fn: Callable[[pd.Series, int], pd.Series],
    folds: int = 6,
    horizon: int = 12,
) -> float:
    clean = series.dropna()
    minimum = max(60, horizon * 3)
    if len(clean) < minimum + horizon:
        return float("nan")

    origins = np.linspace(
        len(clean) - folds * horizon,
        len(clean) - horizon,
        num=folds,
        dtype=int,
    )
    errors = []
    for origin in origins:
        train = clean.iloc[:origin]
        actual = clean.iloc[origin:origin + horizon]
        if len(actual) != horizon:
            continue
        predicted = forecast_fn(train, horizon)
        errors.extend(np.abs(actual.to_numpy() - predicted.to_numpy()))
    return float(np.mean(errors)) if errors else float("nan")


# ---------------------------------------------------------------------
# Original visualizations — retained and extended
# ---------------------------------------------------------------------

def build_chart(
    observed: pd.DataFrame,
    forecast: pd.Series,
    residual_mae: float,
    horizon: int,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=observed.index,
            y=observed["anomaly_c"],
            name="Observed temperature anomaly",
            mode="lines",
            line={"color": BLUE},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast,
            name=f"{horizon}-month forecast",
            mode="lines",
            line={"color": RED, "dash": "dash"},
        )
    )
    if np.isfinite(residual_mae):
        upper = forecast + residual_mae
        lower = forecast - residual_mae
        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=upper,
                name="Approximate MAE band",
                mode="lines",
                line={"width": 0},
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=lower,
                name="Approximate MAE band",
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(214,64,42,0.12)",
                line={"width": 0},
            )
        )
    fig.update_layout(
        yaxis_title="Global surface temperature anomaly (°C)",
        xaxis_title="Month",
        hovermode="x unified",
        legend={"orientation": "h"},
        margin={"l": 20, "r": 20, "t": 25, "b": 20},
    )
    return fig


def build_seasonality_chart(series: pd.Series) -> go.Figure:
    monthly = series.groupby(series.index.month).agg(["mean", "min", "max"])
    labels = [MONTHS[index - 1] for index in monthly.index]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=monthly["max"],
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=monthly["min"],
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(31,95,166,0.12)",
            name="Historical range",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=monthly["mean"],
            mode="lines+markers",
            line={"color": BLUE, "width": 3},
            marker={"size": 7},
            name="Monthly average",
        )
    )
    fig.update_layout(
        yaxis_title="Anomaly (°C)",
        xaxis_title="Calendar month",
        hovermode="x unified",
        margin={"l": 20, "r": 20, "t": 25, "b": 20},
        legend={"orientation": "h"},
    )
    return fig


def build_forecast_table(forecast: pd.Series, latest: float) -> pd.DataFrame:
    table = forecast.rename("Forecast anomaly (°C)").to_frame()
    table["Change vs latest (°C)"] = table["Forecast anomaly (°C)"] - latest
    table["Direction"] = np.where(
        table["Change vs latest (°C)"] > 0.02,
        "Higher",
        np.where(
            table["Change vs latest (°C)"] < -0.02,
            "Lower",
            "Near latest",
        ),
    )
    return table


# ---------------------------------------------------------------------
# Styling and application header
# ---------------------------------------------------------------------

inject_theme()
header(
    "NASA Climate Outlook",
    "Global temperature monitoring, historical exploration, experimental "
    "forecasting and regional climate analysis.",
)


# ---------------------------------------------------------------------
# Sidebar controls and global data
# ---------------------------------------------------------------------

with st.sidebar:
    st.header("Dashboard controls")
    source = st.selectbox(
        "Global dataset",
        ["NASA GISTEMP v4", "NOAA PSL mirror"],
        help="NASA is preferred. NOAA PSL is available as a mirror/fallback.",
    )
    horizon = st.slider("Forecast horizon (months)", 6, 24, 12, 6)
    history_months = st.slider(
        "Chart history (months)",
        60,
        1800,
        360,
        60,
    )
    overview_model = st.selectbox(
        "Overview forecast model",
        ["Auto-select", "Seasonal trend", "Persistence baseline"],
        help=(
            "Auto-select chooses between the original seasonal-trend and "
            "persistence baselines using historical rolling-origin MAE."
        ),
    )

try:
    with st.spinner("Loading global temperature observations..."):
        data = get_global(source)
except Exception as error:
    st.error(f"Could not load global temperature observations: {error}")
    st.stop()

if "anomaly_c" not in data.columns:
    st.error("The selected data source did not return an 'anomaly_c' column.")
    st.stop()

observed = data[["anomaly_c"]].copy()
observed.index = pd.to_datetime(observed.index)
observed = observed[~observed.index.isna()].sort_index()
observed["anomaly_c"] = pd.to_numeric(
    observed["anomaly_c"], errors="coerce"
)
observed = observed.dropna(subset=["anomaly_c"])

if observed.empty:
    st.error("No usable temperature observations were returned.")
    st.stop()

series = observed["anomaly_c"]
source_label = data.attrs.get("source", source)
latest = float(series.iloc[-1])
latest_date = series.index[-1].strftime("%B %Y")

seasonal_mae = rolling_mae(series, seasonal_forecast)
baseline_mae = rolling_mae(series, persistence_forecast)

if overview_model == "Auto-select":
    if np.isfinite(seasonal_mae) and (
        not np.isfinite(baseline_mae) or seasonal_mae <= baseline_mae
    ):
        selected_overview_model = "Seasonal trend"
    else:
        selected_overview_model = "Persistence baseline"
else:
    selected_overview_model = overview_model

overview_forecast_fn = (
    seasonal_forecast
    if selected_overview_model == "Seasonal trend"
    else persistence_forecast
)
overview_forecast = overview_forecast_fn(series, horizon)
overview_mae = (
    seasonal_mae
    if selected_overview_model == "Seasonal trend"
    else baseline_mae
)

trend_window = min(24, len(series))
trend_slope = annualized_slope(series, trend_window)
direction = (
    "Rising"
    if trend_slope > 0.02
    else "Falling"
    if trend_slope < -0.02
    else "Stable"
)
rise_frequency = rising_frequency(series)
forecast_table = build_forecast_table(overview_forecast, latest)
plot_observed = observed.iloc[-history_months:]

warming_stripes(series)


# ---------------------------------------------------------------------
# Main navigation
# ---------------------------------------------------------------------

tab_names = [
    "Overview",
    "Historical Explorer",
    "Forecasting Lab",
    "Model Leaderboard",
    "Regional Analysis",
    "Climate Map",
    "Explainable AI",
    "Datasets & Pipeline",
    "Methodology",
]
tabs = st.tabs(tab_names)


# ---------------------------------------------------------------------
# 1. Overview — original dashboard retained
# ---------------------------------------------------------------------

with tabs[0]:
    metric_columns = st.columns(5)
    metric_columns[0].metric(
        "Latest observation",
        f"{latest:+.2f} °C",
        latest_date,
    )
    metric_columns[1].metric(
        f"Forecast at +{horizon} months",
        f"{overview_forecast.iloc[-1]:+.2f} °C",
    )
    metric_columns[2].metric(
        "Change vs latest",
        f"{overview_forecast.iloc[-1] - latest:+.2f} °C",
    )
    metric_columns[3].metric(
        "Annualized recent trend",
        f"{trend_slope:+.2f} °C/yr",
    )
    metric_columns[4].metric(
        "Direction",
        {"Rising": "▲ ", "Falling": "▼ ", "Stable": "● "}[direction] + direction,
    )

    plot(
        build_chart(plot_observed, overview_forecast, overview_mae, horizon),
        use_container_width=True,
    )
    st.caption(
        f"Data source: {source_label}. "
        f"Latest available observation: {latest_date}. "
        "The shaded band is an approximate historical-error band."
    )

    overview_left, overview_right = st.columns([1.35, 1])
    with overview_left:
        st.subheader("Seasonality by calendar month")
        plot(
            build_seasonality_chart(series.iloc[-min(len(series), 360):]),
            use_container_width=True,
        )

    with overview_right:
        st.subheader("Recent trend signals")
        trend_table = pd.DataFrame(
            {
                "Window": ["12 months", "24 months", "5 years"],
                "Annualized slope (°C/yr)": [
                    annualized_slope(series, min(12, len(series))),
                    annualized_slope(series, min(24, len(series))),
                    annualized_slope(series, min(60, len(series))),
                ],
            }
        ).set_index("Window")
        st.dataframe(
            trend_table.style.format("{:.3f}"),
            use_container_width=True,
        )
        st.metric(
            "Rolling windows trending higher",
            f"{rise_frequency:.0%}"
            if np.isfinite(rise_frequency)
            else "Unavailable",
        )
        st.info(
            f"The latest {trend_window} months are classified as "
            f"**{direction.lower()}**. Longer windows generally provide "
            "a more stable descriptive signal."
        )

    st.subheader("Original baseline model validation")
    validation_columns = st.columns(3)
    validation_columns[0].metric(
        "Selected overview model",
        selected_overview_model,
    )
    validation_columns[1].metric(
        "Selected model MAE",
        f"{overview_mae:.3f} °C"
        if np.isfinite(overview_mae)
        else "Unavailable",
    )
    if np.isfinite(seasonal_mae) and np.isfinite(baseline_mae):
        validation_columns[2].metric(
            "Seasonal MAE vs persistence",
            f"{baseline_mae - seasonal_mae:+.3f} °C",
            help="Positive means seasonal trend has lower historical error.",
        )
    else:
        validation_columns[2].metric(
            "Seasonal MAE vs persistence",
            "Unavailable",
        )

    baseline_comparison = pd.DataFrame(
        {
            "Model": ["Seasonal trend", "Persistence baseline"],
            "MAE (°C)": [seasonal_mae, baseline_mae],
        }
    )
    mae_fig = px.bar(
        baseline_comparison, x="MAE (°C)", y="Model", orientation="h",
        text_auto=".3f", color="Model", color_discrete_sequence=[BLUE, GREY],
    )
    mae_fig.update_layout(showlegend=False, yaxis_title=None)
    plot(mae_fig, height=220)

    st.subheader("Recent observed values")
    st.dataframe(
        observed.tail(12).rename(
            columns={"anomaly_c": "Observed anomaly (°C)"}
        ).style.format("{:.2f}"),
        use_container_width=True,
    )

    st.download_button(
        "Download overview forecast CSV",
        forecast_table.to_csv().encode("utf-8"),
        file_name="nasa_climate_outlook_forecast.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------
# 2. Historical Explorer
# ---------------------------------------------------------------------

with tabs[1]:
    st.subheader("Explore historical temperature anomalies")
    min_year = int(series.index.min().year)
    max_year = int(series.index.max().year)

    if min_year < max_year:
        default_start = max(min_year, max_year - 50)
        year_range = st.slider(
            "Year range",
            min_value=min_year,
            max_value=max_year,
            value=(default_start, max_year),
        )
    else:
        year_range = (min_year, max_year)

    historical = series[
        (series.index.year >= year_range[0])
        & (series.index.year <= year_range[1])
    ]

    aggregation = st.radio(
        "Aggregation",
        ["Monthly", "Annual", "Decadal"],
        horizontal=True,
    )

    if aggregation == "Monthly":
        historical_view = historical
    elif aggregation == "Annual":
        historical_view = historical.groupby(historical.index.year).mean()
        historical_view.index = pd.to_datetime(
            historical_view.index.astype(str) + "-01-01"
        )
    else:
        decades = (historical.index.year // 10) * 10
        historical_view = historical.groupby(decades).mean()
        historical_view.index = pd.to_datetime(
            historical_view.index.astype(str) + "-01-01"
        )

    hist_fig = px.line(
        x=historical_view.index,
        y=historical_view.values,
        labels={"x": "Date", "y": "Temperature anomaly (°C)"},
        title=f"{aggregation} temperature anomalies",
    )
    hist_fig.update_layout(hovermode="x unified")
    plot(hist_fig, use_container_width=True)

    st.subheader("Annual summary")
    st.dataframe(annual_summary(historical), use_container_width=True)

    st.subheader("Record highs and lows")
    st.dataframe(detect_records(historical), use_container_width=True)

    st.download_button(
        "Download selected historical data",
        historical.rename("anomaly_c").to_csv().encode("utf-8"),
        file_name="historical_temperature_anomalies.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------
# 3. Forecasting Lab — seven V2 models
# ---------------------------------------------------------------------

with tabs[2]:
    st.subheader("Experimental forecasting laboratory")
    st.caption(
        "Generate forecasts using statistical and machine-learning models. "
        "Forecasts are extrapolations of the selected historical series."
    )

    forecast_model_name = st.selectbox(
        "Forecasting model",
        MODEL_NAMES,
        key="forecast_lab_model",
    )

    if st.button("Generate forecast", type="primary", key="generate_forecast"):
        try:
            with st.spinner(f"Fitting {forecast_model_name}..."):
                st.session_state["forecast_result"] = forecast_model(
                    series,
                    forecast_model_name,
                    horizon,
                )
                st.session_state["forecast_model_name"] = forecast_model_name
                st.session_state["forecast_source"] = source_label
        except Exception as error:
            st.error(f"Forecast failed: {error}")

    result = st.session_state.get("forecast_result")
    if result is not None:
        if st.session_state.get("forecast_source") != source_label:
            st.warning(
                "The selected dataset has changed since this forecast was "
                "generated. Generate a new forecast to use the current source."
            )
        else:
            fc = result["forecast"]
            fig = go.Figure()
            history = series.iloc[-min(240, len(series)):]
            fig.add_trace(
                go.Scatter(
                    x=history.index,
                    y=history.values,
                    name="Observed",
                    mode="lines",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=fc.index,
                    y=fc.values,
                    name=st.session_state["forecast_model_name"],
                    line={"dash": "dash"},
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=result["upper"].index,
                    y=result["upper"].values,
                    line={"width": 0},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=result["lower"].index,
                    y=result["lower"].values,
                    fill="tonexty",
                    fillcolor="rgba(214,64,42,.15)",
                    line={"width": 0},
                    name="Approximate interval",
                )
            )
            fig.update_layout(
                height=440,
                yaxis_title="Temperature anomaly (°C)",
                xaxis_title="Date",
                hovermode="x unified",
            )
            plot(fig, use_container_width=True)
            st.caption(result["note"])

            forecast_details = pd.DataFrame(
                {
                    "Forecast": result["forecast"],
                    "Lower": result["lower"],
                    "Upper": result["upper"],
                }
            )
            st.dataframe(
                forecast_details.style.format("{:.3f}"),
                use_container_width=True,
            )
            st.download_button(
                "Download model forecast CSV",
                forecast_details.to_csv().encode("utf-8"),
                file_name="nasa_model_forecast.csv",
                mime="text/csv",
            )


# ---------------------------------------------------------------------
# 4. Model Leaderboard
# ---------------------------------------------------------------------

with tabs[3]:
    st.subheader("Model comparison and leaderboard")
    st.caption(
        "Models are evaluated using rolling-origin validation. "
        "This can take several minutes, especially for SARIMA and XGBoost."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        folds = st.slider("Validation folds", 3, 6, 4)
    with col_b:
        validation_horizon = st.selectbox(
            "Validation horizon (months)",
            [6, 12, 18],
            index=1,
        )

    if st.button(
        "Evaluate all models",
        type="primary",
        key="evaluate_all_models",
    ):
        try:
            with st.spinner(
                "Evaluating models. This may take several minutes..."
            ):
                st.session_state["leaderboard"] = evaluate_models(
                    series,
                    MODEL_NAMES,
                    folds,
                    validation_horizon,
                )
                st.session_state["leaderboard_source"] = source_label
        except Exception as error:
            st.error(f"Model evaluation failed: {error}")

    leaderboard = st.session_state.get("leaderboard")
    if leaderboard is not None:
        if st.session_state.get("leaderboard_source") != source_label:
            st.warning(
                "The selected dataset differs from the dataset used for this "
                "leaderboard. Run the evaluation again."
            )
        st.dataframe(
            leaderboard.style.format(
                {
                    "MAE (°C)": "{:.4f}",
                    "RMSE (°C)": "{:.4f}",
                    "MASE": "{:.4f}",
                    "Seconds": "{:.2f}",
                },
                na_rep="—",
            ),
            use_container_width=True,
        )
        valid = leaderboard.dropna(subset=["MAE (°C)"])
        if not valid.empty:
            bar = px.bar(
                valid.sort_values("MAE (°C)"),
                x="Model",
                y="MAE (°C)",
                title="Mean absolute error by model",
            )
            plot(bar, use_container_width=True)

        st.download_button(
            "Download leaderboard CSV",
            leaderboard.to_csv(index=False).encode("utf-8"),
            file_name="nasa_model_leaderboard.csv",
            mime="text/csv",
        )
    else:
        st.info("Run the evaluation to create the model leaderboard.")


# ---------------------------------------------------------------------
# 5. Regional Analysis
# ---------------------------------------------------------------------

with tabs[4]:
    st.subheader("Regional temperature analysis")
    st.caption(
        "Regional averages are calculated from available gridded cells. "
        "They are not interchangeable with the global land-ocean mean."
    )

    regions = {
        "Global": (-90, 90, -180, 180),
        "Arctic": (60, 90, -180, 180),
        "Northern Hemisphere": (0, 90, -180, 180),
        "Southern Hemisphere": (-90, 0, -180, 180),
        "Tropics": (-23.5, 23.5, -180, 180),
        "North America": (15, 75, -170, -50),
        "South America": (-60, 15, -90, -30),
        "Europe": (35, 72, -15, 40),
        "Africa": (-35, 37, -20, 55),
        "Asia": (5, 80, 40, 180),
        "Australia": (-50, -10, 110, 180),
    }

    region_name = st.selectbox("Select region", list(regions))
    if st.button("Load regional series", key="load_regional"):
        try:
            with st.spinner("Loading gridded temperature data..."):
                grid = load_gridded_temperature()
                regional = regional_series(grid, *regions[region_name])
                st.session_state["regional_series"] = regional
                st.session_state["regional_name"] = region_name
        except Exception as error:
            st.error(f"Regional grid unavailable: {error}")

    if "regional_series" in st.session_state:
        regional = st.session_state["regional_series"]
        st.markdown(
            f"**Currently displayed region:** "
            f"{st.session_state.get('regional_name', 'Unknown')}"
        )
        plot(line_fig(regional, "Anomaly (°C)"), height=380)
        st.caption(
            "Missing grid cells are excluded from the regional average."
        )


# ---------------------------------------------------------------------
# 6. Climate Map
# ---------------------------------------------------------------------

with tabs[5]:
    st.subheader("Latest gridded climate anomaly map")
    st.caption(
        "This visualization requires the gridded NASA dataset and may take "
        "time to download on first use."
    )

    if st.button("Load latest climate map", key="load_climate_map"):
        try:
            with st.spinner("Loading gridded climate data..."):
                grid = load_gridded_temperature()
                map_date = pd.Timestamp(grid.time.values[-1])
                map_data = (
                    grid.isel(time=-1)
                    .to_dataframe(name="anomaly_c")
                    .reset_index()
                    .dropna()
                )
                st.session_state["climate_map_data"] = map_data
                st.session_state["climate_map_date"] = map_date
        except Exception as error:
            st.error(f"Climate map unavailable: {error}")

    if "climate_map_data" in st.session_state:
        map_data = st.session_state["climate_map_data"]
        if {"lat", "lon", "anomaly_c"}.issubset(map_data.columns):
            map_fig = px.scatter_geo(
                map_data,
                lat="lat",
                lon="lon",
                color="anomaly_c",
                color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0,
                projection="natural earth",
                title=(
                    "Grid anomaly — "
                    f"{st.session_state['climate_map_date']:%B %Y}"
                ),
            )
            map_fig.update_traces(marker={"size": 4})
            plot(map_fig, use_container_width=True)
        else:
            st.error(
                "The gridded dataset did not contain the expected "
                "lat, lon and anomaly_c fields."
            )


# ---------------------------------------------------------------------
# 7. Explainable AI
# ---------------------------------------------------------------------

with tabs[6]:
    st.subheader("Explainable AI — Random Forest")
    st.write(
        "Inspect how the Random Forest model uses historical lag features "
        "to produce its predictions."
    )
    st.caption(
        "SHAP values describe model feature contributions; they do not "
        "identify causal climate drivers."
    )

    if st.button(
        "Generate Random Forest explanation",
        key="explain_random_forest",
    ):
        try:
            with st.spinner("Training model and calculating SHAP values..."):
                st.session_state["shap_values"] = explain_random_forest(series)
                st.session_state["shap_source"] = source_label
        except Exception as error:
            st.error(f"SHAP explanation unavailable: {error}")

    shap_values = st.session_state.get("shap_values")
    if shap_values is not None:
        if st.session_state.get("shap_source") != source_label:
            st.warning(
                "The selected dataset differs from the dataset used for "
                "this explanation. Generate it again."
            )
        plot(importance_fig(shap_values, "Mean absolute SHAP value"))


# ---------------------------------------------------------------------
# 8. Datasets and pipeline
# ---------------------------------------------------------------------

with tabs[7]:
    st.subheader("Datasets and ingestion pipeline")
    st.markdown(
        """
        **Current data sources**

        - **NASA GISTEMP v4:** global monthly surface-temperature anomalies.
        - **NOAA PSL mirror:** alternative series available when the primary
          NASA source is unavailable or when explicitly selected.
        - **NASA gridded temperature data:** used by the regional analysis
          and climate-map features.

        The NASA and NOAA series can differ in coverage and baseline.
        Their raw values should not be compared directly without
        harmonization.
        """
    )

    st.write("**Active data source:**", source_label)
    st.write("**First observation:**", series.index.min().strftime("%B %Y"))
    st.write("**Latest observation:**", latest_date)
    st.write("**Number of monthly observations:**", f"{len(series):,}")

    st.markdown(
        """
        **Manual data snapshot**

        Run the following command from the project root:

        ```bash
        python ingestion/update_data.py
        ```

        The repository also includes a GitHub Actions workflow intended
        to run this ingestion task daily and commit changed snapshots.
        """
    )

    st.markdown(
        """
        **Future data-source integrations**

        Berkeley Earth and Copernicus ERA5 require separate data adapters
        and baseline/coverage harmonization. They are not silently mixed
        into the current NASA/NOAA series.
        """
    )


# ---------------------------------------------------------------------
# 9. Methodology and limitations
# ---------------------------------------------------------------------

with tabs[8]:
    st.subheader("Methodology and limitations")
    st.markdown(
        """
        ### Observations
        The primary source is NASA GISTEMP v4. NOAA PSL is available as
        an alternative or fallback. The source label identifies which
        dataset supplied the displayed global series.

        ### Forecasting models
        The Forecasting Lab provides seasonal trend, persistence, ARIMA,
        SARIMA, exponential smoothing, Random Forest and XGBoost models.

        ### Validation
        The original overview retains its rolling-origin MAE comparison
        between seasonal trend and persistence. The expanded leaderboard
        evaluates the additional models over rolling historical folds.

        ### Prediction intervals
        ARIMA and SARIMA use model-native intervals. Other models use
        approximate residual-bootstrap intervals. These are not guaranteed
        confidence intervals and are not physical climate projection ranges.

        ### Regional analysis
        Regional averages use gridded observations. Results depend on
        grid coverage, missing cells, coordinate conventions and the
        selected geographical bounds.

        ### Explainable AI
        SHAP describes the contribution of model input features, such as
        historical lag values. It does not establish causal relationships.

        ### Important limitations
        - Short-term statistical extrapolations are not physical climate
          projections or weather forecasts.
        - NASA and NOAA series may differ in coverage and baseline.
        - Model leaderboard rankings depend on the validation period,
          forecast horizon and data source.
        - Results should be interpreted as an educational analytical
          experiment, not an operational climate prediction.
        """
    )
    st.markdown(
        "[NASA GISTEMP](https://data.giss.nasa.gov/gistemp/) · "
        "[NOAA PSL](https://psl.noaa.gov/data/timeseries/)"
    )
