"""
Page 2 — Team Analysis
Head-to-head records, win trends, venue heatmaps, phase batting/bowling.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules.ui_helpers import divider, insight_box, page_header, style_chart


# ---------------------------------------------------------------------------
# Season helper — handles '2007/08', '2008', 2008 alike
# ---------------------------------------------------------------------------

def parse_season(s) -> int:
    """Extract the 4-digit start year from any season format."""
    try:
        return int(str(s)[:4])
    except (ValueError, TypeError):
        return 0


# ---------------------------------------------------------------------------
# Cached aggregations
# ---------------------------------------------------------------------------

@st.cache_data
def _get_teams(df: pd.DataFrame):
    teams = sorted(
        set(df["batting_team"].dropna().unique()) | set(df["bowling_team"].dropna().unique())
    )
    return [t for t in teams if t and str(t) != "nan"]


@st.cache_data
def _h2h(df: pd.DataFrame, t1: str, t2: str) -> dict:
    matches = df.drop_duplicates("match_id")[
        ["match_id", "batting_team", "bowling_team", "match_won_by"]
    ].copy()
    h2h = matches[
        ((matches["batting_team"] == t1) & (matches["bowling_team"] == t2)) |
        ((matches["batting_team"] == t2) & (matches["bowling_team"] == t1))
    ]
    t1_wins = (h2h["match_won_by"] == t1).sum()
    t2_wins = (h2h["match_won_by"] == t2).sum()
    return {"t1_wins": int(t1_wins), "t2_wins": int(t2_wins), "total": len(h2h)}


@st.cache_data
def _win_pct_by_season(df: pd.DataFrame, team: str) -> pd.DataFrame:
    m = df.drop_duplicates("match_id")[
        ["match_id", "season", "batting_team", "bowling_team", "match_won_by"]
    ].copy()
    team_matches = m[(m["batting_team"] == team) | (m["bowling_team"] == team)].copy()
    team_matches["won"] = team_matches["match_won_by"] == team
    result = (
        team_matches.groupby("season")["won"]
        .agg(["sum", "count"])
        .reset_index()
        .assign(**{"Win %": lambda x: (x["sum"] / x["count"] * 100).round(1)})
        .rename(columns={"season": "Season"})
    )
    # Add numeric year for sorting
    result["season_year"] = result["Season"].apply(parse_season)
    return result.sort_values("season_year")


@st.cache_data
def _phase_batting(df: pd.DataFrame, team: str) -> pd.DataFrame:
    t = df[df["batting_team"] == team].copy()
    return (
        t.groupby("phase", observed=True)["runs_total"]
        .sum()
        .reset_index()
        .rename(columns={"runs_total": "Total Runs", "phase": "Phase"})
    )


@st.cache_data
def _venue_win_pct(df: pd.DataFrame, team: str) -> pd.DataFrame:
    m = df.drop_duplicates("match_id")[
        ["match_id", "venue", "batting_team", "bowling_team", "match_won_by"]
    ].copy()
    team_m = m[(m["batting_team"] == team) | (m["bowling_team"] == team)].copy()
    team_m["won"] = team_m["match_won_by"] == team
    vw = team_m.groupby("venue")["won"].agg(["sum", "count"]).reset_index()
    vw["Win %"] = (vw["sum"] / vw["count"] * 100).round(1)
    return vw[vw["count"] >= 3].sort_values("Win %", ascending=False).head(15)


@st.cache_data
def _stage_appearances(df: pd.DataFrame, team: str) -> pd.DataFrame:
    if "stage" not in df.columns:
        return pd.DataFrame()
    m = df.drop_duplicates("match_id")[
        ["match_id", "stage", "batting_team", "bowling_team"]
    ].copy()
    team_m = m[(m["batting_team"] == team) | (m["bowling_team"] == team)]
    return (
        team_m.groupby("stage")["match_id"]
        .count()
        .reset_index()
        .rename(columns={"match_id": "Appearances", "stage": "Stage"})
    )


@st.cache_data
def _economy_by_phase(df: pd.DataFrame, team: str) -> pd.DataFrame:
    """Economy rate conceded by the team as the bowling side, broken by phase."""
    b = df[df["bowling_team"] == team].copy()
    phase_stats = (
        b.groupby("phase", observed=True)
        .agg(runs=("runs_total", "sum"), balls=("valid_ball", "sum"))
        .reset_index()
    )
    phase_stats["Economy"] = (phase_stats["runs"] / (phase_stats["balls"] / 6)).round(2)
    return phase_stats.rename(columns={"phase": "Phase"})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def show(df: pd.DataFrame):
    page_header(
        "🏆", "Team Analysis",
        "Compare any two franchises head-to-head — wins, season trends, "
        "venue performance, and phase-wise batting/bowling strength."
    )

    teams = _get_teams(df)

    # ── Sidebar controls ─────────────────────────────────────────────────────
    st.sidebar.markdown(
        '<hr style="border:none;border-top:1px solid #2A2D35;margin:14px 0"/>',
        unsafe_allow_html=True,
    )
    st.sidebar.subheader("Team Filters")
    t1 = st.sidebar.selectbox("Team 1", teams, index=0, key="ta_t1")
    t2 = st.sidebar.selectbox(
        "Team 2",
        [t for t in teams if t != t1],
        index=min(1, len(teams) - 2),
        key="ta_t2",
    )

    # Season range slider — safe for mixed formats like '2007/08'
    all_seasons = sorted(df["season"].dropna().unique().tolist())
    season_years = sorted({parse_season(s) for s in all_seasons if parse_season(s) > 0})
    min_yr = season_years[0] if season_years else 2008
    max_yr = season_years[-1] if season_years else 2026

    s_range = st.sidebar.slider(
        "Season Range",
        min_value=min_yr,
        max_value=max_yr,
        value=(min_yr, max_yr),
        key="ta_seasons",
    )
    filtered = df[df["season"].apply(lambda s: s_range[0] <= parse_season(s) <= s_range[1])]

    divider()

    # ── Head-to-head ─────────────────────────────────────────────────────────
    with st.spinner("Loading insights…"):
        st.subheader(f"Head-to-Head: {t1} vs {t2}")
        h2h = _h2h(filtered, t1, t2)

    if h2h["total"] == 0:
        st.warning("No head-to-head matches found for the selected seasons.")
    else:
        fig_h2h = go.Figure(
            go.Pie(
                labels=[t1, t2, "No Result"],
                values=[
                    h2h["t1_wins"], h2h["t2_wins"],
                    max(h2h["total"] - h2h["t1_wins"] - h2h["t2_wins"], 0),
                ],
                hole=0.45,
                marker=dict(colors=["#F5A623", "#1C6EF5", "#444"]),
                textinfo="label+percent",
                textfont=dict(size=13),
            )
        )
        style_chart(fig_h2h, f"Head-to-Head ({h2h['total']} matches)")
        st.plotly_chart(fig_h2h, width="stretch")
        insight_box(
            f"<strong>{t1}</strong> has won <strong>{h2h['t1_wins']}</strong> out of "
            f"<strong>{h2h['total']}</strong> encounters against {t2} in the selected period."
        )

    divider()

    # ── Win % by Season ──────────────────────────────────────────────────────
    st.subheader("Win % by Season")
    w1 = _win_pct_by_season(filtered, t1).assign(Team=t1)
    w2 = _win_pct_by_season(filtered, t2).assign(Team=t2)
    win_trend = pd.concat([w1, w2])

    fig_trend = px.line(
        win_trend,
        x="Season", y="Win %", color="Team",
        markers=True,
        color_discrete_map={t1: "#F5A623", t2: "#1C6EF5"},
    )
    fig_trend.update_layout(yaxis=dict(range=[0, 105]))
    fig_trend.update_traces(line_width=2.5, marker_size=8, marker_line_width=0)
    style_chart(fig_trend, "Win Percentage Across Seasons")
    st.plotly_chart(fig_trend, width="stretch")

    divider()

    # ── Phase Batting ─────────────────────────────────────────────────────────
    st.subheader("Batting Strength by Phase")
    col_l, col_r = st.columns(2)
    for col_widget, team, color in [(col_l, t1, "#F5A623"), (col_r, t2, "#1C6EF5")]:
        phase_bat = _phase_batting(filtered, team)
        fig_pb = px.bar(
            phase_bat, x="Phase", y="Total Runs",
            color_discrete_sequence=[color],
        )
        style_chart(fig_pb, f"{team} — Runs by Phase")
        col_widget.plotly_chart(fig_pb, width="stretch")

    divider()

    # ── Economy by Phase ─────────────────────────────────────────────────────
    st.subheader("Economy Rate Conceded by Phase")
    col_l2, col_r2 = st.columns(2)
    for col_widget, team, color in [(col_l2, t1, "#F5A623"), (col_r2, t2, "#1C6EF5")]:
        econ = _economy_by_phase(filtered, team)
        fig_ec = px.bar(
            econ, x="Phase", y="Economy",
            color_discrete_sequence=[color],
            text="Economy",
        )
        fig_ec.update_traces(texttemplate="%{text:.2f}", textposition="outside",
                              marker_line_width=0)
        style_chart(fig_ec, f"{team} — Economy by Phase")
        col_widget.plotly_chart(fig_ec, width="stretch")

    divider()

    # ── Venue Win % ──────────────────────────────────────────────────────────
    st.subheader(f"Venue Win % — {t1}")
    venue_df = _venue_win_pct(filtered, t1)
    if not venue_df.empty:
        fig_v = px.bar(
            venue_df,
            x="Win %", y="venue",
            orientation="h",
            color="Win %",
            color_continuous_scale="Oranges",
            hover_data={"count": True},
        )
        fig_v.update_layout(
            coloraxis_showscale=False,
            yaxis=dict(categoryorder="total ascending"),
        )
        style_chart(fig_v, f"{t1} Win % by Venue (min 3 matches)")
        st.plotly_chart(fig_v, width="stretch")

    divider()

    # ── Stage Appearances ────────────────────────────────────────────────────
    st.subheader(f"Finals & Playoff Appearances — {t1}")
    stage_df = _stage_appearances(filtered, t1)
    if not stage_df.empty:
        fig_stage = px.bar(
            stage_df, x="Stage", y="Appearances",
            color_discrete_sequence=["#F5A623"],
            text="Appearances",
        )
        fig_stage.update_traces(textposition="outside", marker_line_width=0)
        style_chart(fig_stage, f"{t1} — Match Count by Stage")
        st.plotly_chart(fig_stage, width="stretch")
    else:
        insight_box("Stage column not available in the dataset.")
