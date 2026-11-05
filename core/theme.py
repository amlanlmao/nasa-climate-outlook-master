"""Visual theme for the NASA Climate Outlook dashboard.

Drop-in replacement for core/theme.py. Keeps the existing public API:
inject_theme, header, warming_stripes, style_fig, plot, line_fig,
and importance_fig.
"""
from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import sample_colorscale


# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

BLUE = "#2878B8"
RED = "#D95D50"
TEAL = "#2A9D8F"
GREY = "#94A3B8"
INK = "#172B3A"
MUTED = "#718096"
HAIR = "rgba(148,163,184,.25)"

FONT = "'Public Sans', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
DISPLAY = "'Bricolage Grotesque', 'Public Sans', system-ui, sans-serif"

COLORWAY = [BLUE, RED, TEAL, "#8A70B5", "#D49A45", GREY]


# ---------------------------------------------------------------------------
# Streamlit CSS
# ---------------------------------------------------------------------------

_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700;12..96,800&family=Public+Sans:wght@400;500;600;700&display=swap');

:root {{
  --climate-blue: {BLUE};
  --climate-red: {RED};
  --climate-teal: {TEAL};
  --climate-hair: {HAIR};
}}

.stApp, .stApp p, .stApp li, .stApp label, .stApp button,
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stCaptionContainer"] {{
  font-family: {FONT};
}}

.block-container {{
  max-width: 1380px;
  padding-top: 2.15rem;
  padding-bottom: 4rem;
}}

[data-testid="stHeader"] {{
  background: transparent;
}}

h1, h2, h3, .app-title, [data-testid="stMetricValue"] {{
  font-family: {DISPLAY} !important;
  letter-spacing: -0.035em;
}}

h1 {{
  font-size: clamp(2rem, 4vw, 2.8rem);
  line-height: 1.08;
}}
h2, h3 {{
  margin-top: 1.45rem;
  letter-spacing: -0.025em;
}}
h2 {{ font-size: 1.45rem; }}
h3 {{ font-size: 1.12rem; }}

.app-title {{
  font-size: clamp(2.15rem, 4.3vw, 3.15rem);
  font-weight: 700;
  line-height: 1.03;
  margin: 0;
}}
.app-sub {{
  opacity: .72;
  max-width: 68ch;
  margin: .65rem 0 1.35rem;
  font-size: .98rem;
  line-height: 1.6;
}}

/* Metrics: editorial top rule, no heavy cards or gradients. */
[data-testid="stMetric"] {{
  background: transparent;
  border: 0;
  border-top: 2px solid rgba(148,163,184,.48);
  border-radius: 0;
  padding: .8rem .15rem .35rem;
  min-height: 104px;
}}
[data-testid="stMetricLabel"] p {{
  font-size: .82rem;
  font-weight: 600;
  opacity: .76;
}}
[data-testid="stMetricValue"] {{
  font-size: clamp(1.45rem, 2.2vw, 1.95rem);
  font-weight: 700;
}}
[data-testid="stMetricDelta"] {{
  font-size: .78rem;
}}

/* Tabs and controls. */
.stTabs [data-baseweb="tab-list"] {{
  gap: clamp(.75rem, 2vw, 1.55rem);
  border-bottom: 1px solid var(--climate-hair);
}}
.stTabs [data-baseweb="tab"] {{
  padding: .7rem 0;
}}
.stTabs [data-baseweb="tab"] p {{
  font-weight: 600;
  font-size: .9rem;
}}
.stTabs [data-baseweb="tab"][aria-selected="true"] {{
  border-bottom-color: var(--climate-blue);
}}
.stButton > button, .stDownloadButton > button {{
  border-radius: 8px;
  font-weight: 650;
  min-height: 2.55rem;
  transition: transform .15s ease, box-shadow .15s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
  transform: translateY(-1px);
  box-shadow: 0 5px 16px rgba(40,120,184,.12);
}}
[data-testid="stSelectbox"] [data-baseweb="select"],
[data-testid="stMultiSelect"] [data-baseweb="select"] {{
  border-radius: 8px;
}}

/* Tables, alerts and sidebar. */
[data-testid="stDataFrame"], [data-testid="stTable"] {{
  border: 1px solid var(--climate-hair);
  border-radius: 10px;
  overflow: hidden;
}}
[data-testid="stAlert"] {{
  border-radius: 10px;
}}
[data-testid="stExpander"] {{
  border: 1px solid var(--climate-hair);
  border-radius: 10px;
}}
[data-testid="stSidebar"] {{
  border-right: 1px solid var(--climate-hair);
}}
[data-testid="stSidebar"] h2 {{
  font-size: 1.12rem;
}}

/* Warming stripes. */
.stripes {{
  display: flex;
  height: 100px;
  border-radius: 7px;
  overflow: hidden;
  isolation: isolate;
  box-shadow: 0 1px 0 rgba(148,163,184,.15);
}}
.stripes i {{
  flex: 1 1 0;
  min-width: 1px;
  transition: transform .12s ease, filter .12s ease;
}}
.stripes i:hover {{
  filter: brightness(1.12);
  transform: scaleY(1.035);
  z-index: 1;
}}
.stripes-note {{
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  font-size: .78rem;
  opacity: .68;
  margin: .48rem 0 1.45rem;
}}

/* Small visual separators used by the dashboard. */
.climate-divider {{
  height: 1px;
  background: var(--climate-hair);
  margin: 1.2rem 0;
}}
.climate-eyebrow {{
  color: var(--climate-blue);
  font-size: .72rem;
  font-weight: 700;
  letter-spacing: .12em;
  text-transform: uppercase;
}}

@media (max-width: 800px) {{
  .block-container {{
    padding-top: 1.35rem;
    padding-left: 1rem;
    padding-right: 1rem;
  }}
  .stripes {{ height: 76px; }}
  .stTabs [data-baseweb="tab-list"] {{ gap: .8rem; }}
  .stTabs [data-baseweb="tab"] p {{ font-size: .82rem; }}
}}
@media (max-width: 540px) {{
  .app-title {{ font-size: 2rem; }}
  .stripes {{ height: 64px; }}
  .stripes-note span:nth-child(2) {{ display: none; }}
}}
"""


def inject_theme() -> None:
    """Inject CSS. Call once, immediately after st.set_page_config()."""
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


def header(title: str, subtitle: str) -> None:
    """Render the dashboard masthead using the shared type scale."""
    safe_title = escape(str(title))
    safe_subtitle = escape(str(subtitle))
    st.markdown(
        f'<div class="app-title">{safe_title}</div>'
        f'<p class="app-sub">{safe_subtitle}</p>',
        unsafe_allow_html=True,
    )


def warming_stripes(series: pd.Series) -> None:
    """Render one hoverable stripe per year, coloured by annual mean anomaly."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        st.info("Warming stripes are unavailable because no valid observations were found.")
        return

    annual = clean.groupby(clean.index.year).mean().sort_index()
    if annual.empty:
        return

    # Use a symmetric scale around zero so blue and red have a stable meaning.
    span = max(float(annual.abs().max()), 0.5)
    normalized = np.clip((annual.to_numpy(dtype=float) / span + 1.0) / 2.0, 0.0, 1.0)
    colors = sample_colorscale("RdBu_r", normalized.tolist())

    cells = "".join(
        f'<i style="background:{color}" title="{int(year)}: {value:+.2f} °C"></i>'
        for year, value, color in zip(annual.index, annual.to_numpy(), colors)
    )
    st.markdown(
        f'<div class="stripes" role="img" aria-label="Annual temperature anomaly warming stripes">{cells}</div>'
        f'<div class="stripes-note"><span>{int(annual.index[0])}</span>'
        "<span>Each stripe represents one year's mean temperature anomaly</span>"
        f"<span>{int(annual.index[-1])}</span></div>",
        unsafe_allow_html=True,
    )


def style_fig(fig: go.Figure, height: int | None = None) -> go.Figure:
    """Apply the common Plotly style without changing chart data."""
    has_title = bool(fig.layout.title.text)
    has_legend = any(trace.showlegend is not False and trace.name for trace in fig.data)
    top = 58 if has_title else (46 if has_legend else 16)

    fig.update_layout(
        colorway=COLORWAY,
        font={"family": FONT, "size": 12, "color": "#64748B"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 14, "r": 14, "t": top, "b": 14},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "title_text": "",
            "bgcolor": "rgba(0,0,0,0)",
        },
        hoverlabel={
            "font_family": FONT,
            "bgcolor": "#172B3A",
            "font": {"color": "#FFFFFF"},
            "bordercolor": "#172B3A",
        },
        title={
            "x": 0,
            "xanchor": "left",
            "font": {"size": 17, "family": DISPLAY, "color": "#64748B"},
        },
        hovermode="x unified",
    )

    is_geo = any(getattr(trace, "type", "") in {"scattergeo", "choropleth"} for trace in fig.data)
    if is_geo:
        fig.update_geos(
            bgcolor="rgba(0,0,0,0)",
            showframe=False,
            showcoastlines=True,
            coastlinecolor="rgba(148,163,184,.58)",
            showland=True,
            landcolor="rgba(148,163,184,.14)",
            showcountries=True,
            countrycolor="rgba(148,163,184,.22)",
            showocean=False,
        )
        fig.update_coloraxes(
            colorbar={
                "title": {"text": "°C"},
                "thickness": 11,
                "len": 0.72,
                "outlinewidth": 0,
                "ticks": "outside",
            }
        )
    else:
        fig.update_xaxes(
            showgrid=False,
            linecolor=HAIR,
            ticks="outside",
            tickcolor=HAIR,
            automargin=True,
            zeroline=False,
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor="rgba(148,163,184,.20)",
            zeroline=True,
            zerolinecolor="rgba(148,163,184,.42)",
            automargin=True,
        )

    if height is not None:
        fig.update_layout(height=height)
    return fig


def plot(fig, height=None, **kwargs):
    kwargs.pop("use_container_width", None)

    st.plotly_chart(
        style_fig(fig, height),
        use_container_width=True,
        config={"displaylogo": False, "responsive": True},
        **kwargs,
    )


def line_fig(data, ylabel: str) -> go.Figure:
    """Create a consistent line chart from a Series or one-column DataFrame."""
    series = data.to_series() if hasattr(data, "to_series") else data
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    fig = go.Figure(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            line={"color": BLUE, "width": 2.2},
            fill="tozeroy",
            fillcolor="rgba(40,120,184,.07)",
            hovertemplate="%{x|%b %Y}<br>%{y:.2f} °C<extra></extra>",
        )
    )
    fig.update_layout(
        yaxis_title=ylabel,
        xaxis_title="Date",
        hovermode="x unified",
    )
    return fig


def importance_fig(values, xlabel: str) -> go.Figure:
    """Create a horizontal feature-importance chart from a Series or DataFrame."""
    series = values.iloc[:, 0] if isinstance(values, pd.DataFrame) else pd.Series(values)
    series = pd.to_numeric(series, errors="coerce").dropna().sort_values()
    fig = go.Figure(
        go.Bar(
            x=series.values,
            y=[str(index) for index in series.index],
            orientation="h",
            marker={"color": TEAL, "line": {"width": 0}},
            hovertemplate="%{y}<br>%{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=None,
        height=max(320, 34 * len(series)),
        margin={"l": 130, "r": 14, "t": 42, "b": 42},
    )
    return fig
