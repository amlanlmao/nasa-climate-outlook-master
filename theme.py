"""Visual theme for the NASA Climate Outlook dashboard (CSS, Plotly style, small components)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import sample_colorscale

BLUE, RED, TEAL, GREY = "#1f5fa6", "#d6402a", "#2a9d8f", "#94a3b8"
FONT = "'Public Sans', system-ui, sans-serif"
DISPLAY = "'Bricolage Grotesque', 'Public Sans', sans-serif"
HAIR = "rgba(148,163,184,.30)"

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700&family=Public+Sans:wght@400;500;600&display=swap');

.stApp, .stApp p, .stApp li, .stApp label, .stApp button,
.stApp [data-testid="stMarkdownContainer"] { font-family: 'Public Sans', system-ui, sans-serif; }
.block-container { max-width: 1320px; padding-top: 2.4rem; }

h1, h2, h3, .app-title, [data-testid="stMetricValue"] {
  font-family: 'Bricolage Grotesque', 'Public Sans', sans-serif !important;
  letter-spacing: -0.02em;
}
h2, h3 { margin-top: 1.6rem; }

.app-title { font-size: 2.7rem; font-weight: 700; line-height: 1.05; margin: 0; }
.app-sub   { opacity: .72; max-width: 62ch; margin: .65rem 0 1.5rem; font-size: 1.02rem; line-height: 1.55; }

.stripes { display: flex; height: 108px; border-radius: 6px; overflow: hidden; }
.stripes i { flex: 1 1 0; }
.stripes-note { display: flex; justify-content: space-between; gap: 1rem;
  font-size: .82rem; opacity: .7; margin: .5rem 0 1.6rem; }

[data-testid="stMetric"] { background: none; border: 0; border-top: 2px solid rgba(148,163,184,.45);
  border-radius: 0; padding: .75rem 0 .25rem; }
[data-testid="stMetricLabel"] p { font-size: .86rem; opacity: .75; }
[data-testid="stMetricValue"] { font-size: 1.9rem; font-weight: 700; }

.stTabs [data-baseweb="tab-list"] { gap: 1.5rem; }
.stTabs [data-baseweb="tab"] { padding: .65rem 0; }
.stTabs [data-baseweb="tab"] p { font-weight: 600; font-size: .95rem; }

.stButton > button, .stDownloadButton > button { border-radius: 6px; font-weight: 600; }
[data-testid="stDataFrame"] { border: 1px solid rgba(148,163,184,.3); border-radius: 6px; overflow: hidden; }
[data-testid="stAlert"] { border-radius: 6px; }
[data-testid="stSidebar"] { border-right: 1px solid rgba(148,163,184,.25); }
[data-testid="stSidebar"] h2 { font-size: 1.15rem; }

@media (max-width: 640px) {
  .app-title { font-size: 2rem; }
  .stripes { height: 72px; }
  .stripes-note span:nth-child(2) { display: none; }
}
"""


def inject_theme() -> None:
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


def header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="app-title">{title}</div><p class="app-sub">{subtitle}</p>',
        unsafe_allow_html=True,
    )


def warming_stripes(series: pd.Series) -> None:
    """One stripe per year, coloured by that year's mean anomaly."""
    annual = series.groupby(series.index.year).mean()
    span = max(float(annual.abs().max()), 0.5)
    colors = sample_colorscale("RdBu_r", [(v / span + 1) / 2 for v in annual])
    cells = "".join(
        f'<i style="background:{c}" title="{y}: {v:+.2f} °C"></i>'
        for y, v, c in zip(annual.index, annual, colors)
    )
    st.markdown(
        f'<div class="stripes">{cells}</div>'
        f'<div class="stripes-note"><span>{annual.index[0]}</span>'
        "<span>Each stripe is one year’s average temperature anomaly</span>"
        f"<span>{annual.index[-1]}</span></div>",
        unsafe_allow_html=True,
    )


def style_fig(fig: go.Figure, height: int | None = None) -> go.Figure:
    has_title = bool(fig.layout.title.text)
    has_legend = any(t.showlegend is not False and t.name for t in fig.data)
    top = 60 if has_title else (44 if has_legend else 12)
    fig.update_layout(
        colorway=[BLUE, RED, TEAL, GREY],
        font={"family": FONT, "size": 13},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 8, "r": 8, "t": top, "b": 8},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02,
                "xanchor": "left", "x": 0, "title_text": ""},
        hoverlabel={"font_family": FONT},
        title={"x": 0, "xanchor": "left", "font": {"size": 17, "family": DISPLAY}},
    )
    if any(t.type == "scattergeo" for t in fig.data):
        fig.update_geos(
            bgcolor="rgba(0,0,0,0)", showframe=False, showcoastlines=True,
            coastlinecolor="rgba(148,163,184,.7)", showland=True,
            landcolor="rgba(148,163,184,.14)", showocean=False,
        )
        fig.update_coloraxes(colorbar={"title": "°C", "thickness": 12, "len": 0.7})
    else:
        fig.update_xaxes(showgrid=False, linecolor=HAIR, ticks="outside",
                         tickcolor=HAIR, automargin=True)
        fig.update_yaxes(gridcolor="rgba(148,163,184,.22)", zeroline=True,
                         zerolinecolor="rgba(148,163,184,.55)", automargin=True)
    if height:
        fig.update_layout(height=height)
    return fig


def plot(fig: go.Figure, height: int | None = None, **_) -> None:
    st.plotly_chart(style_fig(fig, height), use_container_width=True)


def line_fig(data, ylabel: str) -> go.Figure:
    s = data.to_series() if hasattr(data, "to_series") else data
    s = s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s
    fig = go.Figure(go.Scatter(x=s.index, y=s.values, mode="lines", line={"width": 1.8}))
    fig.update_layout(yaxis_title=ylabel, hovermode="x unified")
    return fig


def importance_fig(values, xlabel: str) -> go.Figure:
    s = values.iloc[:, 0] if isinstance(values, pd.DataFrame) else pd.Series(values)
    s = s.sort_values()
    fig = go.Figure(go.Bar(x=s.values, y=[str(i) for i in s.index],
                           orientation="h", marker_color=TEAL))
    fig.update_layout(xaxis_title=xlabel)
    return fig
