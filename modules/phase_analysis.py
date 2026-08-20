"""
Page 4 — Phase & Situational Analysis
The differentiator page: powerplay deep-dives, death-over specialists,
chase vs defend, DRS success rates, and superover drama.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules.ui_helpers import divider, insight_box, page_header, style_chart


# ---------------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------------

@st.cache_data
def _run_rate_by_phase(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df[df["valid_ball"] == 1]
        .groupby(["batting_team", "phase"], observed=True)
        .agg(runs=("runs_total", "sum"), balls=("valid_ball", "count"))
        .reset_index()
    )
    agg["Run Rate"] = (agg["runs"] / (agg["balls"] / 6)).round(2)
    return agg


@st.cache_data
def _wicket_by_phase(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[df["is_wicket"] == True]
        .groupby("phase", observed=True)["is_wicket"]
        .count()
        .reset_index()
        .rename(columns={"is_wicket": "Wickets", "phase": "Phase"})
    )


@st.cache_data
def _dot_pct_by_phase_season(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[df["valid_ball"] == 1]
    agg = (
        valid.groupby(["season", "phase"], observed=True)
        .agg(dots=("is_dot", "sum"), balls=("valid_ball", "count"))
        .reset_index()
    )
    agg["Dot %"] = (agg["dots"] / agg["balls"] * 100).round(2)
    return agg


@st.cache_data
def _best_pp_teams(df: pd.DataFrame) -> pd.DataFrame:
    pp = df[(df["over"] <= 5) & (df["innings"] == 1)].copy()
    match_pp = pp.groupby(["match_id", "batting_team"])["runs_total"].sum().reset_index()
    return (
        match_pp.groupby("batting_team")["runs_total"]
        .mean()
        .reset_index()
        .rename(columns={"runs_total": "Avg PP Runs", "batting_team": "Team"})
        .sort_values("Avg PP Runs", ascending=False)
    )


@st.cache_data
def _best_pp_bowlers(df: pd.DataFrame, min_matches: int = 20) -> pd.DataFrame:
    pp = df[(df["over"] <= 5) & (df["valid_ball"] == 1)].copy()
    agg = (
        pp.groupby("bowler")
        .agg(runs=("runs_total", "sum"), balls=("valid_ball", "count"),
             matches=("match_id", "nunique"))
        .reset_index()
    )
    agg["Economy"] = (agg["runs"] / (agg["balls"] / 6)).round(2)
    return (
        agg[agg["matches"] >= min_matches]
        .sort_values("Economy")
        .head(10)
        .rename(columns={"bowler": "Bowler"})
    )


@st.cache_data
def _death_batters(df: pd.DataFrame, min_balls: int = 100) -> pd.DataFrame:
    death = df[(df["over"] >= 15) & (df["valid_ball"] == 1)].copy()
    agg = (
        death.groupby("batter")
        .agg(runs=("runs_batter", "sum"), balls=("valid_ball", "count"))
        .reset_index()
    )
    agg["Strike Rate"] = (agg["runs"] / agg["balls"] * 100).round(2)
    return (
        agg[agg["balls"] >= min_balls]
        .sort_values("Strike Rate", ascending=False)
        .head(10)
        .rename(columns={"batter": "Batter"})
    )


@st.cache_data
def _chase_defend_trend(df: pd.DataFrame) -> pd.DataFrame:
    m = df.drop_duplicates("match_id")[
        ["match_id", "season", "innings", "batting_team", "match_won_by"]
    ].copy()
    first = m[m["innings"] == 1][["match_id", "season", "batting_team"]].rename(
        columns={"batting_team": "first_bat"}
    )
    last = m.drop_duplicates("match_id")[["match_id", "match_won_by"]].copy()
    merged = first.merge(last, on="match_id")
    merged["batting_first_won"] = merged["first_bat"] == merged["match_won_by"]
    trend = (
        merged.groupby("season")["batting_first_won"]
        .agg(["sum", "count"])
        .reset_index()
    )
    trend["Bat First Win %"] = (trend["sum"] / trend["count"] * 100).round(1)
    trend["Chase Win %"] = (100 - trend["Bat First Win %"]).round(1)
    return trend


@st.cache_data
def _venue_chase_pct(df: pd.DataFrame) -> pd.DataFrame:
    m = df.drop_duplicates("match_id")[
        ["match_id", "venue", "innings", "batting_team", "match_won_by"]
    ].copy()
    first = m[m["innings"] == 1][["match_id", "venue", "batting_team"]].rename(
        columns={"batting_team": "first_bat"}
    )
    result = m.drop_duplicates("match_id")[["match_id", "match_won_by"]]
    merged = first.merge(result, on="match_id")
    merged["batting_first_won"] = merged["first_bat"] == merged["match_won_by"]
    agg = (
        merged.groupby("venue")["batting_first_won"]
        .agg(["sum", "count"])
        .reset_index()
    )
    agg["Chase Win %"] = ((1 - agg["sum"] / agg["count"]) * 100).round(1)
    return agg[agg["count"] >= 5].sort_values("Chase Win %", ascending=False).head(15)


@st.cache_data
def _drs_stats(df: pd.DataFrame) -> pd.DataFrame:
    if "review_decision" not in df.columns:
        return pd.DataFrame()
    rev = df[df["review_decision"].notna()].copy()
    agg = (
        rev.groupby("batting_team")["review_decision"]
        .apply(
            lambda s: pd.Series({
                "total": len(s),
                "upheld": (s == "upheld").sum(),
            })
        )
        .unstack()
        .reset_index()
    )
    agg.columns = ["Team", "Total Reviews", "Upheld"]
    agg["Success %"] = (agg["Upheld"] / agg["Total Reviews"] * 100).round(1)
    return agg.sort_values("Success %", ascending=False)


@st.cache_data
def _superover_stats(df: pd.DataFrame):
    if "superover_winner" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    so = df[df["superover_winner"].notna()].drop_duplicates("match_id")
    per_season = so.groupby("season")["match_id"].count().reset_index().rename(
        columns={"match_id": "Superovers"}
    )
    team_count = pd.concat([so["batting_team"], so["bowling_team"]]).value_counts().reset_index()
    team_count.columns = ["Team", "Appearances"]
    return per_season, team_count.head(10)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def show(df: pd.DataFrame):
    page_header(
        "📈", "Phase & Situational Analysis",
        "Powerplay deep dives, death-over specialists, chase vs defend trends, "
        "DRS intelligence, and superover drama."
    )

    seasons = sorted(df["season"].dropna().unique().tolist())
    sel_seasons = st.sidebar.multiselect(
        "Season(s)", seasons, default=seasons, key="pa_seasons"
    )
    if not sel_seasons:
        sel_seasons = seasons
    filtered = df[df["season"].isin(sel_seasons)]

    divider()

    # ══════════════════════════════════════════════════════════════════
    # SECTION 1 — Phase Breakdown
    # ══════════════════════════════════════════════════════════════════
    st.subheader("📊 Phase Breakdown")

    with st.spinner("Loading insights…"):
        rr = _run_rate_by_phase(filtered)
        top_teams = (
            filtered.groupby("batting_team")["runs_total"].sum()
            .nlargest(8).index.tolist()
        )

    rr_top = rr[rr["batting_team"].isin(top_teams)]
    fig_rr = px.bar(
        rr_top, x="batting_team", y="Run Rate", color="phase",
        barmode="group",
        color_discrete_sequence=["#F5A623", "#1C6EF5", "#E74C3C"],
    )
    fig_rr.update_layout(xaxis_title="Team", legend_title="Phase")
    style_chart(fig_rr, "Run Rate by Phase (Top 8 Teams)")
    st.plotly_chart(fig_rr, width="stretch")

    wkt_phase = _wicket_by_phase(filtered)
    fig_wkt = px.bar(
        wkt_phase, x="Phase", y="Wickets", color="Phase",
        color_discrete_sequence=["#F5A623", "#1C6EF5", "#E74C3C"],
        text="Wickets",
    )
    fig_wkt.update_traces(textposition="outside", marker_line_width=0)
    fig_wkt.update_layout(showlegend=False)
    style_chart(fig_wkt, "Wickets Fallen by Phase")
    st.plotly_chart(fig_wkt, width="stretch")

    dot_df = _dot_pct_by_phase_season(filtered)
    fig_dot = px.line(
        dot_df, x="season", y="Dot %", color="phase",
        markers=True,
        color_discrete_sequence=["#F5A623", "#1C6EF5", "#E74C3C"],
    )
    fig_dot.update_layout(xaxis=dict(tickmode="linear"), legend_title="Phase")
    fig_dot.update_traces(line_width=2.5, marker_size=7, marker_line_width=0)
    style_chart(fig_dot, "Dot Ball % by Phase Across Seasons")
    st.plotly_chart(fig_dot, width="stretch")
    insight_box(
        "Death-over dot ball % has been <strong>declining</strong> season-over-season "
        "as batters have become more aggressive in the final overs."
    )

    divider()

    # ══════════════════════════════════════════════════════════════════
    # SECTION 2 — Powerplay Deep Dive
    # ══════════════════════════════════════════════════════════════════
    st.subheader("⚡ Powerplay Deep Dive (Overs 0–5)")
    col1, col2 = st.columns(2)

    pp_teams = _best_pp_teams(filtered)
    fig_ppt = px.bar(
        pp_teams.head(10), x="Avg PP Runs", y="Team",
        orientation="h",
        color="Avg PP Runs", color_continuous_scale="Oranges",
    )
    fig_ppt.update_layout(
        yaxis=dict(categoryorder="total ascending"),
        coloraxis_showscale=False,
    )
    style_chart(fig_ppt, "Best Powerplay Teams (Avg Runs)")
    col1.plotly_chart(fig_ppt, width="stretch")

    pp_bowl = _best_pp_bowlers(filtered)
    fig_ppb = px.bar(
        pp_bowl, x="Economy", y="Bowler",
        orientation="h",
        color="Economy", color_continuous_scale="Blues_r",
    )
    fig_ppb.update_layout(
        yaxis=dict(categoryorder="total descending"),
        coloraxis_showscale=False,
    )
    style_chart(fig_ppb, "Best Powerplay Bowlers (Economy, min 20 matches)")
    col2.plotly_chart(fig_ppb, width="stretch")

    divider()

    # ══════════════════════════════════════════════════════════════════
    # SECTION 3 — Death Over Specialists
    # ══════════════════════════════════════════════════════════════════
    st.subheader("☠️ Death Over Specialists (Overs 15–19)")
    col3, col4 = st.columns(2)

    db = _death_batters(filtered)
    fig_db = px.bar(
        db, x="Strike Rate", y="Batter",
        orientation="h",
        color="Strike Rate", color_continuous_scale="Oranges",
        hover_data={"runs": True, "balls": True},
    )
    fig_db.update_layout(
        yaxis=dict(categoryorder="total ascending"),
        coloraxis_showscale=False,
    )
    style_chart(fig_db, "Best Death Batters (SR, min 100 balls)")
    col3.plotly_chart(fig_db, width="stretch")

    death_bowl = filtered[(filtered["over"] >= 15) & (filtered["valid_ball"] == 1)]
    da = (
        death_bowl.groupby("bowler")
        .agg(runs=("runs_total", "sum"), balls=("valid_ball", "count"))
        .reset_index()
    )
    da["Economy"] = (da["runs"] / (da["balls"] / 6)).round(2)
    da = da[da["balls"] >= 60].sort_values("Economy").head(10)
    fig_da = px.bar(
        da, x="Economy", y="bowler",
        orientation="h",
        color="Economy", color_continuous_scale="Blues_r",
    )
    fig_da.update_layout(
        yaxis=dict(categoryorder="total descending"),
        coloraxis_showscale=False,
    )
    style_chart(fig_da, "Best Death Bowlers (Economy, min 60 balls)")
    col4.plotly_chart(fig_da, width="stretch")

    divider()

    # ══════════════════════════════════════════════════════════════════
    # SECTION 4 — Chase vs Defend
    # ══════════════════════════════════════════════════════════════════
    st.subheader("🎯 Chase vs Defend Analysis")

    cd = _chase_defend_trend(filtered)
    fig_cd = go.Figure()
    fig_cd.add_trace(go.Scatter(
        x=cd["season"], y=cd["Bat First Win %"],
        name="Batting First Win %", mode="lines+markers",
        line=dict(color="#F5A623", width=2.5), marker=dict(size=8),
    ))
    fig_cd.add_trace(go.Scatter(
        x=cd["season"], y=cd["Chase Win %"],
        name="Chasing Win %", mode="lines+markers",
        line=dict(color="#1C6EF5", width=2.5), marker=dict(size=8),
    ))
    fig_cd.update_layout(
        xaxis=dict(tickmode="linear"),
        yaxis=dict(range=[0, 100]),
    )
    style_chart(fig_cd, "Batting First vs Chasing Win % Across Seasons")
    st.plotly_chart(fig_cd, width="stretch")

    overall_first = cd["Bat First Win %"].mean().round(1)
    overall_chase = cd["Chase Win %"].mean().round(1)
    insight_box(
        f"Across selected seasons, teams <strong>batting first</strong> win "
        f"<strong>{overall_first}%</strong> of matches on average, while "
        f"<strong>chasing teams</strong> win <strong>{overall_chase}%</strong>."
    )

    venue_chase = _venue_chase_pct(filtered)
    if not venue_chase.empty:
        fig_vc = px.scatter(
            venue_chase,
            x="count", y="Chase Win %",
            hover_name="venue",
            size="count", color="Chase Win %",
            color_continuous_scale="Oranges",
        )
        fig_vc.add_hline(y=50, line_dash="dash", line_color="#444", opacity=0.6)
        style_chart(fig_vc, "Chase Win % by Venue (bubble = total matches)")
        st.plotly_chart(fig_vc, width="stretch")

    divider()

    # ══════════════════════════════════════════════════════════════════
    # SECTION 5 — DRS & Reviews
    # ══════════════════════════════════════════════════════════════════
    st.subheader("🔍 DRS & Review Intelligence")
    drs = _drs_stats(filtered)
    if drs.empty:
        insight_box("DRS / review_decision data not found in the dataset.")
    else:
        fig_drs = px.bar(
            drs.head(10), x="Success %", y="Team",
            orientation="h",
            color="Success %", color_continuous_scale="Greens",
            hover_data={"Total Reviews": True, "Upheld": True},
        )
        fig_drs.update_layout(
            yaxis=dict(categoryorder="total ascending"),
            coloraxis_showscale=False,
        )
        style_chart(fig_drs, "DRS Success Rate by Team (% of Reviews Upheld)")
        st.plotly_chart(fig_drs, width="stretch")
        insight_box(
            "Teams with higher DRS success rates tend to have better analytical "
            "support staff reviewing LBW and edge decisions."
        )

    divider()

    # ══════════════════════════════════════════════════════════════════
    # SECTION 6 — Superover Drama
    # ══════════════════════════════════════════════════════════════════
    st.subheader("⚡ Superover Drama")
    so_season, so_teams = _superover_stats(filtered)

    if so_season.empty:
        insight_box("Superover data not available in the dataset.")
    else:
        col5, col6 = st.columns(2)

        fig_so_s = px.bar(
            so_season, x="season", y="Superovers",
            color_discrete_sequence=["#E74C3C"],
        )
        style_chart(fig_so_s, "Superovers Per Season")
        col5.plotly_chart(fig_so_s, width="stretch")

        if not so_teams.empty:
            fig_so_t = px.bar(
                so_teams, x="Appearances", y="Team",
                orientation="h",
                color_discrete_sequence=["#E74C3C"],
            )
            fig_so_t.update_layout(yaxis=dict(categoryorder="total ascending"))
            style_chart(fig_so_t, "Teams Most Involved in Superovers")
            col6.plotly_chart(fig_so_t, width="stretch")
