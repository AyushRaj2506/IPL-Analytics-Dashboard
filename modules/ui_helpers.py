"""
Shared UI helper utilities — page headers, insight boxes, animated KPI cards,
and a chart-style function applied to every Plotly figure.
"""
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# Chart styling — call after building any Plotly figure
# ---------------------------------------------------------------------------

CHART_FONT = "Inter, sans-serif"

def style_chart(fig: go.Figure, title: str = None, height: int = None) -> go.Figure:
    """Apply consistent dark-theme, Inter font, and animation to any figure."""
    updates = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=CHART_FONT, color="white", size=13),
        hoverlabel=dict(bgcolor="#1C1F26", font_size=13, font_family=CHART_FONT),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="white")),
    )
    if title:
        updates["title"] = dict(text=title, font=dict(size=18, color="#F5A623"), x=0)
    if height:
        updates["height"] = height
    fig.update_layout(**updates)
    fig.update_traces(marker_line_width=0)
    return fig


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

def page_header(icon: str, title: str, subtitle: str):
    """Render a styled page header with orange underline."""
    st.markdown(
        f"""
        <div style="border-bottom:2px solid #F5A623; padding-bottom:12px; margin-bottom:24px">
          <h1 style="color:white; margin:0; font-family:'Inter',sans-serif;
                     font-size:2.2rem; font-weight:700; letter-spacing:-0.5px">
            {icon}&nbsp;{title}
          </h1>
          <p style="color:#888; margin:6px 0 0 0; font-size:0.95rem;
                    font-family:'Inter',sans-serif; line-height:1.5">
            {subtitle}
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Animated KPI card
# ---------------------------------------------------------------------------

def kpi_card(col, label: str, value: str, sub: str = ""):
    """Render an animated card-style metric inside a column."""
    with col:
        st.markdown(
            f"""
            <div style="background:#1C1F26; border-left:4px solid #F5A623;
                        border-radius:10px; padding:20px 20px 16px 20px;
                        margin:4px 0; box-shadow:0 4px 16px rgba(0,0,0,0.5);
                        transition:transform 0.2s ease;">
              <p style="color:#aaa; font-size:0.78rem; margin:0;
                        text-transform:uppercase; letter-spacing:0.08em;
                        font-family:'Inter',sans-serif">{label}</p>
              <h2 style="color:#F5A623; font-size:2rem; margin:6px 0 2px 0;
                         font-weight:700; font-family:'Inter',sans-serif">{value}</h2>
              <p style="color:#666; font-size:0.75rem; margin:0;
                        font-family:'Inter',sans-serif">{sub}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Insight box
# ---------------------------------------------------------------------------

def insight_box(text: str):
    """Render a styled blue-accent insight callout instead of st.info()."""
    st.markdown(
        f"""
        <div style="background:#0e1d30; border-left:4px solid #4A9EFF;
                    border-radius:8px; padding:16px 20px; margin:14px 0;
                    font-family:'Inter',sans-serif">
          <span style="color:#4A9EFF; font-weight:600">💡 Insight&nbsp;&nbsp;</span>
          <span style="color:#ccc; font-size:0.93rem; line-height:1.6">{text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Section divider
# ---------------------------------------------------------------------------

def divider():
    """Styled thin-line section divider."""
    st.markdown(
        '<hr style="border:none; border-top:1px solid #2A2D35; margin:24px 0"/>',
        unsafe_allow_html=True,
    )
