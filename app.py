import streamlit as st
from modules.data_loader import load_data
from modules import overview, team_analysis, player_intel, phase_analysis, win_predictor

st.set_page_config(
    page_title="IPL Analytics Dashboard",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Inter font ─────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1 { font-size: 2.2rem; font-weight: 700; letter-spacing: -0.5px; }
    h2 { font-size: 1.5rem; font-weight: 600; }
    h3 { font-size: 1.15rem; font-weight: 600; }
    p, div { font-size: 0.95rem; line-height: 1.6; }

    /* ── Sidebar ─────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #1C1F26 100%);
        border-right: 1px solid #2A2D35;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 1rem;
        padding: 6px 0;
        color: #ccc;
        font-family: 'Inter', sans-serif;
    }
    [data-testid="stSidebar"] .stRadio [data-testid="stMarkdown"] {
        font-size: 0.85rem;
    }

    /* ── Native metric cards (fallback) ──────────────────────────── */
    [data-testid="metric-container"] {
        background: #1C1F26;
        border: 1px solid #2A2D35;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }
    [data-testid="metric-container"] label {
        font-size: 0.78rem !important;
        color: #888 !important;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        font-family: 'Inter', sans-serif !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #F5A623 !important;
    }

    /* ── Plotly chart container ──────────────────────────────────── */
    .stPlotlyChart {
        border-radius: 12px;
        overflow: hidden;
        background: rgba(28,31,38,0.6) !important;
        padding: 4px;
    }

    /* ── Alert / st.info overrides ───────────────────────────────── */
    .stAlert {
        border-radius: 8px;
        font-family: 'Inter', sans-serif;
    }

    /* ── Selectbox / slider labels ───────────────────────────────── */
    .stSelectbox label, .stMultiSelect label, .stSlider label {
        font-size: 0.85rem !important;
        color: #aaa !important;
    }

    /* ── Subheader styling ───────────────────────────────────────── */
    [data-testid="stMarkdownContainer"] h3 {
        color: #e0e0e0;
        margin-top: 8px;
    }

    /* ── Scrollbar ───────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0E1117; }
    ::-webkit-scrollbar-thumb { background: #2A2D35; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #F5A623; }

    /* ── Hide Streamlit chrome ───────────────────────────────────── */
    [data-testid="stToolbar"] { display: none !important; }
    header[data-testid="stHeader"] { background: rgba(0,0,0,0); }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Load data ────────────────────────────────────────────────────────────────
with st.spinner("Loading ball-by-ball IPL data (295K+ records)…"):
    df = load_data()

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.markdown(
    """
    <p style="font-size:1.15rem; font-weight:700; color:white;
              margin:8px 0 2px 0; font-family:'Inter',sans-serif">
      IPL Analytics
    </p>
    <p style="font-size:0.8rem; color:#888; margin:0;
              font-family:'Inter',sans-serif">
      2008 – 2026 &nbsp;·&nbsp; 1,243 Matches
    </p>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    '<hr style="border:none; border-top:1px solid #2A2D35; margin:14px 0"/>',
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigate",
    [
        "📊 Season Overview",
        "🏆 Team Analysis",
        "🎯 Player Intelligence",
        "📈 Phase Analysis",
        "🤖 Win Predictor",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown(
    '<hr style="border:none; border-top:1px solid #2A2D35; margin:14px 0"/>',
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    "<small style='color:#555; font-family:Inter,sans-serif'>"
    "Built with Python · Streamlit · XGBoost · Plotly"
    "</small>",
    unsafe_allow_html=True,
)

# ── Route pages ──────────────────────────────────────────────────────────────
if page == "📊 Season Overview":
    overview.show(df)
elif page == "🏆 Team Analysis":
    team_analysis.show(df)
elif page == "🎯 Player Intelligence":
    player_intel.show(df)
elif page == "📈 Phase Analysis":
    phase_analysis.show(df)
elif page == "🤖 Win Predictor":
    win_predictor.show(df)
