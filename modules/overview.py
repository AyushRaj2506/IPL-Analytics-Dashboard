"""
Page 1 — Season Overview
Shows season-level KPIs, run trends, toss impact, and top venues.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules.ui_helpers import divider, insight_box, kpi_card, page_header, style_chart


# ---------------------------------------------------------------------------
# Cached aggregations
# ---------------------------------------------------------------------------

@st.cache_data
def _season_runs(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("season")["runs_total"]
        .sum()
        .reset_index()
        .rename(columns={"runs_total": "Total Runs"})
        .sort_values("season")
    )


@st.cache_data
def _avg_first_innings_score(df: pd.DataFrame) -> pd.DataFrame:
    first = df[df["innings"] == 1]
    match_scores = first.groupby(["season", "match_id"])["runs_total"].sum().reset_index()
    return (
        match_scores.groupby("season")["runs_total"]
        .mean()
        .reset_index()
        .rename(columns={"runs_total": "Avg 1st Innings Score"})
        .sort_values("season")
    )


@st.cache_data
def _toss_impact(df: pd.DataFrame) -> pd.DataFrame:
    match_level = df.drop_duplicates("match_id")[
        ["match_id", "toss_winner", "toss_decision", "match_won_by"]
    ].copy()
    match_level["toss_win_match_win"] = (
        match_level["toss_winner"] == match_level["match_won_by"]
    )
    summary = (
        match_level.groupby("toss_decision")["toss_win_match_win"]
        .agg(["sum", "count"])
        .reset_index()
    )
    summary["Win %"] = (summary["sum"] / summary["count"] * 100).round(1)
    summary.rename(columns={"toss_decision": "Toss Decision"}, inplace=True)
    return summary


@st.cache_data
def _top_venues(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return (
        df.drop_duplicates("match_id")
        .groupby("venue")["match_id"]
        .count()
        .reset_index()
        .rename(columns={"match_id": "Matches Hosted"})
        .sort_values("Matches Hosted", ascending=False)
        .head(n)
    )


@st.cache_data
def _potm_leader(df: pd.DataFrame) -> str:
    if "player_of_match" not in df.columns:
        return "N/A"
    counts = df.drop_duplicates("match_id")["player_of_match"].value_counts()
    if counts.empty:
        return "N/A"
    return f"{counts.idxmax()} ({counts.max()})"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def show(df: pd.DataFrame):
    page_header(
        "📊", "Season Overview",
        "High-level statistics across every IPL season — runs, wickets, "
        "toss impact, and venue distribution."
    )

    # ── Sidebar season filter ────────────────────────────────────────────────
    seasons = sorted(df["season"].dropna().unique().tolist())
    selected = st.sidebar.multiselect(
        "Filter by Season", seasons, default=seasons, key="overview_seasons"
    )
    if not selected:
        selected = seasons
    filtered = df[df["season"].isin(selected)]

    divider()

    # ── KPI Cards ────────────────────────────────────────────────────────────
    with st.spinner("Loading insights…"):
        total_matches = filtered["match_id"].nunique()
        total_runs = int(filtered["runs_total"].sum())
        total_wickets = int(filtered["is_wicket"].sum())
        potm = _potm_leader(filtered)

    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "🏟️ Total Matches", f"{total_matches:,}", f"{len(selected)} seasons")
    kpi_card(c2, "🏏 Total Runs", f"{total_runs:,}", "All innings combined")
    kpi_card(c3, "💥 Total Wickets", f"{total_wickets:,}", "Confirmed dismissals")
    kpi_card(c4, "🏆 Most POTM Awards", potm, "Player of the Match")

    divider()

    # ── Runs per Season ──────────────────────────────────────────────────────
    st.subheader("Runs per Season")
    runs_df = _season_runs(filtered)
    fig_runs = px.bar(
        runs_df,
        x="season", y="Total Runs",
        color="Total Runs",
        color_continuous_scale="Oranges",
        labels={"season": "Season", "Total Runs": "Total Runs"},
    )
    fig_runs.update_layout(
        coloraxis_showscale=False,
        xaxis=dict(tickmode="linear"),
    )
    style_chart(fig_runs, "Total Runs Scored Each Season")
    st.plotly_chart(fig_runs, width="stretch")

    # ── Avg First Innings Score ──────────────────────────────────────────────
    st.subheader("Average First Innings Score by Season")
    avg_df = _avg_first_innings_score(filtered)
    fig_avg = px.line(
        avg_df,
        x="season", y="Avg 1st Innings Score",
        markers=True,
        color_discrete_sequence=["#F5A623"],
    )
    fig_avg.update_layout(xaxis=dict(tickmode="linear"))
    fig_avg.update_traces(line_width=2.5, marker_size=8, marker_line_width=0)
    style_chart(fig_avg, "Average First Innings Score per Season")
    st.plotly_chart(fig_avg, width="stretch")

    divider()

    # ── Toss Impact ──────────────────────────────────────────────────────────
    st.subheader("Toss Impact on Match Result")
    toss_df = _toss_impact(filtered)
    fig_toss = px.bar(
        toss_df,
        x="Toss Decision", y="Win %",
        color="Toss Decision",
        color_discrete_sequence=["#F5A623", "#1C6EF5"],
        text="Win %",
    )
    fig_toss.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                            marker_line_width=0)
    fig_toss.update_layout(showlegend=False, yaxis=dict(range=[0, 105]))
    style_chart(fig_toss, "Win % for Toss Winner by Decision (Bat vs Field)")
    st.plotly_chart(fig_toss, width="stretch")

    if len(toss_df) >= 2:
        bat_row = toss_df[toss_df["Toss Decision"] == "bat"]
        field_row = toss_df[toss_df["Toss Decision"] == "field"]
        bat_pct = bat_row["Win %"].values[0] if not bat_row.empty else "?"
        field_pct = field_row["Win %"].values[0] if not field_row.empty else "?"
        insight_box(
            f"Teams choosing to <strong>field</strong> after winning the toss won "
            f"<strong>{field_pct}%</strong> of those matches, vs <strong>{bat_pct}%</strong> "
            f"when batting first — confirming that fielding-first is the dominant IPL strategy."
        )

    divider()

    # ── Top Venues ───────────────────────────────────────────────────────────
    st.subheader("Top 10 Venues by Matches Hosted")
    venue_df = _top_venues(filtered)
    fig_venue = px.bar(
        venue_df,
        x="Matches Hosted", y="venue",
        orientation="h",
        color="Matches Hosted",
        color_continuous_scale="Oranges",
    )
    fig_venue.update_layout(
        coloraxis_showscale=False,
        yaxis=dict(categoryorder="total ascending"),
    )
    style_chart(fig_venue, "Most Frequently Used IPL Venues")
    st.plotly_chart(fig_venue, width="stretch")

    insight_box(
        "Wankhede (Mumbai) and Chinnaswamy (Bengaluru) are among the most-used IPL venues, "
        "reflecting their strong infrastructure and large crowd capacities."
    )
