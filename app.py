from __future__ import annotations

from io import StringIO
from time import sleep
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


NASA_GISTEMP_URL = (
    "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
)
NASA_GISTEMP_MIRROR_URL = (
    "https://psl.noaa.gov/data/timeseries/month/data/nasa.land.csv"
)
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@st.cache_data(ttl=3600, show_spinner=False)
def load_gistemp() -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1):
        try:
            response = requests.get(
                NASA_GISTEMP_URL,
                headers={"User-Agent": "NASAClimateOutlook/0.1"},
                timeout=3,
            )
            response.raise_for_status()
            raw = pd.read_csv(StringIO(response.text), skiprows=1)
            raw = raw.rename(columns={raw.columns[0]: "Year"})
            raw["Year"] = pd.to_numeric(raw["Year"], errors="coerce")
            raw = raw.dropna(subset=["Year"])
            rows = []
            for _, row in raw.iterrows():
                for month_number, month in enumerate(MONTHS, start=1):
                    value = pd.to_numeric(row.get(month), errors="coerce")
                    if pd.notna(value):
                        rows.append(
                            {
                                "date": pd.Timestamp(int(row["Year"]), month_number, 1),
                                "anomaly_c": float(value),
                            }
                        )
            result = pd.DataFrame(rows).sort_values("date").set_index("date")
            if not result.empty:
                result.attrs["source"] = "NASA GISTEMP v4 land-ocean"
                return result
            raise ValueError("NASA returned no usable temperature observations.")
        except (requests.RequestException, ValueError, KeyError) as error:
            last_error = error
            if attempt == 2:
                break
            sleep(1)
    try:
        mirror = requests.get(
            NASA_GISTEMP_MIRROR_URL,
            headers={"User-Agent": "NASAClimateOutlook/0.1"},
            timeout=5,
        )
        mirror.raise_for_status()
        fallback = pd.read_csv(
            StringIO(mirror.text),
            skiprows=1,
            names=["date", "anomaly_c"],
        )
        fallback["date"] = pd.to_datetime(fallback["date"], errors="coerce")
        fallback["anomaly_c"] = pd.to_numeric(fallback["anomaly_c"], errors="coerce")
        fallback = fallback[
            fallback["anomaly_c"].between(-20, 20)
        ].dropna().set_index("date").sort_index()
        if fallback.empty:
            raise ValueError("NASA mirror returned no usable temperature observations.")
        fallback.attrs["source"] = "NOAA PSL mirror of NASA GISTEMP land series"
        return fallback
    except (requests.RequestException, ValueError, KeyError) as mirror_error:
        raise requests.RequestException(
            f"NASA GISTEMP and its NOAA mirror were unavailable. "
            f"Direct error: {last_error}; mirror error: {mirror_error}"
        ) from mirror_error


def seasonal_forecast(series: pd.Series, horizon: int) -> pd.Series:
    """Forecast each calendar month from its recent seasonal mean and trend."""
    clean = series.dropna().astype(float)
    if len(clean) < 36:
        raise ValueError("At least 36 monthly observations are required.")
    recent = clean.iloc[-min(len(clean), 180) :]
    month_means = recent.groupby(recent.index.month).mean()
    month_index = np.arange(1, 13)
    available = month_means.reindex(month_index).interpolate().bfill().ffill()
    trend_window = min(len(recent), 120)
    x = np.arange(trend_window)
    slope = float(np.polyfit(x, recent.iloc[-trend_window:].to_numpy(), 1)[0])
    last_date = clean.index[-1]
    dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    steps = np.arange(1, horizon + 1)
    values = np.array([available.loc[date.month] for date in dates]) + slope * steps
    return pd.Series(values, index=dates, name="forecast_c")


def persistence_forecast(series: pd.Series, horizon: int) -> pd.Series:
    clean = series.dropna().astype(float)
    if clean.empty:
        raise ValueError("At least one observation is required.")
    dates = pd.date_range(
        clean.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"
    )
    return pd.Series(clean.iloc[-1], index=dates, name="forecast_c")


def annualized_slope(series: pd.Series, months: int) -> float:
    clean = series.dropna().astype(float).iloc[-months:]
    if len(clean) < 3:
        return float("nan")
    return float(np.polyfit(np.arange(len(clean)), clean.to_numpy(), 1)[0] * 12)


def rising_frequency(series: pd.Series, months: int = 24, window: int = 12) -> float:
    clean = series.dropna().astype(float)
    if len(clean) < months + window:
        return float("nan")
    slopes = []
    for end in range(months, len(clean) + 1):
        sample = clean.iloc[end - months : end]
        slopes.append(annualized_slope(sample, window))
    return float(np.mean(np.array(slopes) > 0.02))


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
        actual = clean.iloc[origin : origin + horizon]
        predicted = forecast_fn(train, horizon)
        errors.extend(np.abs(actual.to_numpy() - predicted.to_numpy()))
    return float(np.mean(errors))


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
            name="NASA observed",
            mode="lines",
            line={"color": "#2563eb"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast,
            name=f"{horizon}-month forecast",
            mode="lines",
            line={"color": "#dc2626", "dash": "dash"},
        )
    )
    if np.isfinite(residual_mae):
        upper = forecast + residual_mae
        lower = forecast - residual_mae
        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=upper,
                name="Approx. MAE band",
                mode="lines",
                line={"width": 0},
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=lower,
                name="Approx. MAE band",
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(220,38,38,0.12)",
                line={"width": 0},
                showlegend=True,
            )
        )
    fig.update_layout(
        yaxis_title="Global surface temperature anomaly (°C)",
        xaxis_title="Month",
        hovermode="x unified",
        legend={"orientation": "h"},
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
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
            fillcolor="rgba(37,99,235,0.12)",
            name="Historical range",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=monthly["mean"],
            mode="lines+markers",
            line={"color": "#2563eb", "width": 3},
            marker={"size": 7},
            name="Monthly average",
        )
    )
    fig.update_layout(
        yaxis_title="Anomaly (°C)",
        xaxis_title="Calendar month",
        hovermode="x unified",
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        legend={"orientation": "h"},
    )
    return fig


def build_forecast_table(forecast: pd.Series, latest: float) -> pd.DataFrame:
    table = forecast.rename("Forecast anomaly (°C)").to_frame()
    table["Change vs latest (°C)"] = table["Forecast anomaly (°C)"] - latest
    table["Direction"] = np.where(
        table["Change vs latest (°C)"] > 0.02,
        "Higher",
        np.where(table["Change vs latest (°C)"] < -0.02, "Lower", "Near latest"),
    )
    return table


st.set_page_config(page_title="NASA Climate Outlook", page_icon="🌍", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(37,99,235,.12), rgba(15,23,42,.02));
        border: 1px solid rgba(148,163,184,.25);
        padding: 14px;
        border-radius: 12px;
    }
    [data-testid="stMetricValue"] { font-size: 1.65rem; }
    .section-label {
        color: #64748b; font-size: .82rem; font-weight: 700;
        letter-spacing: .08em; text-transform: uppercase;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("NASA Climate Outlook")
st.caption("Explore observed global temperature anomalies, model quality, and a short-term outlook.")

try:
    data = load_gistemp()
except (requests.RequestException, ValueError, KeyError) as error:
    st.error(f"Could not load NASA GISTEMP data: {error}")
    st.stop()
    raise RuntimeError(f"Could not load NASA GISTEMP data: {error}") from error

observed = data.loc[data.index <= data.index.max()]
source_label = data.attrs.get("source", "NASA climate dataset")

with st.sidebar:
    st.header("Forecast controls")
    horizon = st.slider("Forecast horizon (months)", 6, 24, 24, 6)
    history_months = st.slider("Chart history (months)", 60, 360, 180, 12)
    model_name = st.selectbox(
        "Forecast model",
        ["Auto-select", "Seasonal trend", "Persistence baseline"],
        help="Seasonal trend uses recent calendar-month averages and a linear trend. "
        "Persistence repeats the latest observation. Auto-select chooses the lower "
        "historical MAE.",
    )

series = observed["anomaly_c"]
seasonal_mae = rolling_mae(series, seasonal_forecast)
baseline_mae = rolling_mae(series, persistence_forecast)
if model_name == "Auto-select":
    if np.isfinite(seasonal_mae) and (
        not np.isfinite(baseline_mae) or seasonal_mae <= baseline_mae
    ):
        model_name = "Seasonal trend"
    else:
        model_name = "Persistence baseline"
model = seasonal_forecast if model_name == "Seasonal trend" else persistence_forecast
forecast = model(series, horizon=horizon)
model_mae = seasonal_mae if model_name == "Seasonal trend" else baseline_mae
plot_observed = observed.iloc[-history_months:]

latest = observed.iloc[-1]["anomaly_c"]
latest_date = observed.index[-1].strftime("%B %Y")
change = float(forecast.iloc[-1] - latest)
trend_window = min(24, len(series))
trend_slope = annualized_slope(series, trend_window)
direction = "Rising" if trend_slope > 0.02 else "Falling" if trend_slope < -0.02 else "Stable"
rise_frequency = rising_frequency(series)
forecast_table = build_forecast_table(forecast, latest)

st.markdown('<div class="section-label">Current conditions and outlook</div>', unsafe_allow_html=True)
metric_columns = st.columns(5)
metric_columns[0].metric("Latest NASA observation", f"{latest:.2f} °C", latest_date)
metric_columns[1].metric(f"Forecast at +{horizon} months", f"{forecast.iloc[-1]:.2f} °C")
metric_columns[2].metric("Change vs latest", f"{change:+.2f} °C")
metric_columns[3].metric("Annualized recent trend", f"{trend_slope:+.2f} °C/yr")
metric_columns[4].metric("Direction", direction)

st.plotly_chart(
    build_chart(plot_observed, forecast, model_mae, horizon),
    use_container_width=True,
)
st.caption(f"Data source: {source_label}. Latest available observation: {latest_date}.")

tab_overview, tab_models, tab_forecast = st.tabs(
    ["Climate patterns", "Model validation", "Forecast details"]
)

with tab_overview:
    overview_columns = st.columns([1.35, 1])
    with overview_columns[0]:
        st.subheader("Seasonality by calendar month")
        st.plotly_chart(
            build_seasonality_chart(series.iloc[-min(len(series), 360) :]),
            use_container_width=True,
        )
    with overview_columns[1]:
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
            trend_table.style.format("{:.2f}"),
            use_container_width=True,
        )
        st.metric(
            "Rolling windows trending higher",
            f"{rise_frequency:.0%}" if np.isfinite(rise_frequency) else "Unavailable",
        )
        st.info(
            f"The latest {trend_window} months are classified as **{direction.lower()}**. "
            "Use the longer windows for a more stable signal."
        )

with tab_models:
    st.subheader("Model validation")
    st.caption("Historical rolling-origin validation uses six 12-month forecast windows.")
    validation_columns = st.columns(3)
    validation_columns[0].metric("Selected model", model_name)
    validation_columns[1].metric(f"{model_name} MAE", f"{model_mae:.2f} °C")
    if np.isfinite(model_mae) and np.isfinite(baseline_mae):
        validation_columns[2].metric(
            "Seasonal MAE vs persistence",
            f"{(baseline_mae - seasonal_mae):+.2f} °C",
            help="Positive means seasonal trend has lower historical error.",
        )
    else:
        validation_columns[2].metric("Seasonal MAE vs persistence", "Unavailable")

    comparison = pd.DataFrame(
        {
            "Model": ["Seasonal trend", "Persistence baseline"],
            "MAE (°C)": [seasonal_mae, baseline_mae],
            "Selected": [
                model_name == "Seasonal trend",
                model_name == "Persistence baseline",
            ],
        }
    )
    st.bar_chart(comparison.set_index("Model")["MAE (°C)"], color="#2563eb")
    st.dataframe(comparison.style.format({"MAE (°C)": "{:.2f}"}), use_container_width=True)

with tab_forecast:
    st.subheader("Forecast details")
    st.dataframe(
        forecast_table.style.format(
            {
                "Forecast anomaly (°C)": "{:.2f}",
                "Change vs latest (°C)": "{:+.2f}",
            }
        ),
        use_container_width=True,
    )
    st.download_button(
        "Download forecast CSV",
        forecast_table.to_csv().encode("utf-8"),
        file_name="nasa_climate_outlook_forecast.csv",
        mime="text/csv",
    )

with st.expander("Method and limitations"):
    st.markdown(
        """
        - **Observed data:** NASA GISTEMP v4 global land-ocean monthly anomalies, relative to NASA's
          1951–1980 baseline.
        - **Forecast:** selectable seasonal-trend model or persistence baseline.
        - **Validation:** rolling-origin 12-month mean absolute error on historical observations.
        - **Trend signal:** linear slope over the latest 24 observations, annualized. It is a
          descriptive signal, not a causal climate projection.
        - The shaded region is an approximate historical-error band, not a formal confidence interval.
          This is an analytical outlook, not a climate projection or weather prediction.
        - Refreshes are cached for one hour. Validate the latest data timestamp before operational use.
        """
    )

st.subheader("Forecast values")
st.dataframe(
    observed.tail(12).rename(columns={"anomaly_c": "Observed anomaly (°C)"}).style.format("{:.2f}"),
    use_container_width=True,
)
