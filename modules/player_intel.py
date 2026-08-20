"""
Page 3 — Player Intelligence
Batting / bowling scatter plots, consistency charts, and bat position analysis.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules.ui_helpers import divider, insight_box, page_header, style_chart


# ---------------------------------------------------------------------------
# Batting aggregation
# ---------------------------------------------------------------------------

@st.cache_data
def _batting_stats(df: pd.DataFrame, min_innings: int) -> pd.DataFrame:
    valid = df[df["valid_ball"] == 1].copy()

    runs     = valid.groupby("batter")["runs_batter"].sum().rename("runs")
    balls    = valid.groupby("batter")["valid_ball"].count().rename("balls")
    fours    = valid.groupby("batter")["is_four"].sum().rename("fours")
    sixes    = valid.groupby("batter")["is_six"].sum().rename("sixes")

    wickets = (
        df[df["is_wicket"] == True]
        .groupby("batter")["is_wicket"]
        .sum()
        .rename("dismissals")
    )
    innings = (
        valid.groupby("batter")[["match_id", "innings"]]
        .apply(lambda x: x.drop_duplicates().shape[0])
        .rename("innings")
    )

    stats = pd.concat([runs, balls, fours, sixes, wickets, innings], axis=1).dropna(
        subset=["runs", "balls"]
    )
    stats["dismissals"] = stats["dismissals"].fillna(0)
    stats["Strike Rate"] = (stats["runs"] / stats["balls"] * 100).round(2)
    stats["Average"] = stats.apply(
        lambda r: r["runs"] / r["dismissals"] if r["dismissals"] > 0 else r["runs"],
        axis=1,
    ).round(2)
    stats["Boundary %"] = ((stats["fours"] + stats["sixes"]) / stats["balls"] * 100).round(2)

    stats = stats.reset_index().rename(columns={"batter": "Player"})
    return stats[stats["innings"] >= min_innings]


# ---------------------------------------------------------------------------
# Bowling aggregation
# ---------------------------------------------------------------------------

@st.cache_data
def _bowling_stats(df: pd.DataFrame, min_matches: int) -> pd.DataFrame:
    valid = df[df["valid_ball"] == 1].copy()

    runs    = valid.groupby("bowler")["runs_total"].sum().rename("runs_conceded")
    balls   = valid.groupby("bowler")["valid_ball"].count().rename("balls")
    matches = valid.groupby("bowler")["match_id"].nunique().rename("matches")

    not_bowler = ["run out", "obstructing the field", "retired hurt", "retired out"]
    wkt_df = df[(df["striker_out"] == 1) & (df["wicket_kind"].notna())].copy()
    wkt_df = wkt_df[~wkt_df["wicket_kind"].isin(not_bowler)]
    wickets = wkt_df.groupby("bowler")["striker_out"].sum().rename("wickets")

    # walrus-free concat
    stats = pd.concat([runs, balls, wickets, matches], axis=1)
    stats["wickets"] = stats["wickets"].fillna(0)
    stats["Economy"] = (stats["runs_conceded"] / (stats["balls"] / 6)).round(2)
    stats["Bowling SR"] = stats.apply(
        lambda r: r["balls"] / r["wickets"] if r["wickets"] > 0 else None, axis=1
    ).round(2)
    stats["Bowling Avg"] = stats.apply(
        lambda r: r["runs_conceded"] / r["wickets"] if r["wickets"] > 0 else None, axis=1
    ).round(2)

    stats = stats.reset_index().rename(columns={"bowler": "Player"})
    return stats[stats["matches"] >= min_matches]


# ---------------------------------------------------------------------------
# Per-player helpers
# ---------------------------------------------------------------------------

@st.cache_data
def _innings_scores(df: pd.DataFrame, player: str) -> pd.DataFrame:
    p = df[(df["batter"] == player) & (df["valid_ball"] == 1)]
    scores = (
        p.groupby(["match_id", "innings"])["runs_batter"]
        .sum()
        .reset_index()
        .rename(columns={"runs_batter": "Runs"})
        .sort_values(["match_id", "innings"])
    )
    scores["Innings #"] = range(1, len(scores) + 1)
    return scores


@st.cache_data
def _player_phase_runs(df: pd.DataFrame, player: str) -> pd.DataFrame:
    p = df[(df["batter"] == player) & (df["valid_ball"] == 1)]
    return (
        p.groupby("phase", observed=True)["runs_batter"]
        .sum()
        .reset_index()
        .rename(columns={"runs_batter": "Runs", "phase": "Phase"})
    )


@st.cache_data
def _bat_pos_stats(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[(df["valid_ball"] == 1) & (df["bat_pos"].notna())].copy()
    agg = (
        valid.groupby("bat_pos")
        .agg(runs=("runs_batter", "sum"), balls=("valid_ball", "count"))
        .reset_index()
    )
    agg["Strike Rate"] = (agg["runs"] / agg["balls"] * 100).round(2)
    return agg.rename(columns={"bat_pos": "Bat Position", "runs": "Total Runs"})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def show(df: pd.DataFrame):
    page_header(
        "🎯", "Player Intelligence",
        "Deep-dive into batter and bowler metrics — efficiency, consistency, "
        "phase contributions, and positional analysis."
    )

    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.markdown(
        '<hr style="border:none;border-top:1px solid #2A2D35;margin:14px 0"/>',
        unsafe_allow_html=True,
    )
    st.sidebar.subheader("Player Filters")
    role = st.sidebar.radio("Role", ["Batters", "Bowlers"], key="pi_role")
    seasons = sorted(df["season"].dropna().unique().tolist())
    sel_seasons = st.sidebar.multiselect(
        "Season(s)", seasons, default=seasons, key="pi_seasons"
    )
    if not sel_seasons:
        sel_seasons = seasons
    min_innings = st.sidebar.slider("Min Innings / Matches", 5, 50, 15, key="pi_min")

    filtered = df[df["season"].isin(sel_seasons)]
    divider()

    # ═══════════════════════════════════════════════════════════════════════
    #  BATTING
    # ═══════════════════════════════════════════════════════════════════════
    if role == "Batters":
        st.subheader("🏏 Batting Analysis")
        with st.spinner("Loading insights…"):
            bat = _batting_stats(filtered, min_innings)

        if bat.empty:
            st.warning("No data for the selected filters.")
            return

        # SR vs Average scatter
        st.markdown("#### Strike Rate vs Batting Average")
        top10 = bat.nlargest(10, "runs")["Player"].tolist()
        bat["Label"] = bat["Player"].apply(lambda p: p if p in top10 else "")

        fig_scatter = px.scatter(
            bat,
            x="Average", y="Strike Rate",
            size="runs", color="Boundary %",
            hover_name="Player",
            hover_data={"runs": True, "balls": True, "innings": True},
            color_continuous_scale="Oranges",
        )
        fig_scatter.add_hline(y=130, line_dash="dash", line_color="#444", opacity=0.7)
        fig_scatter.add_vline(x=30,  line_dash="dash", line_color="#444", opacity=0.7)
        for _, row in bat[bat["Label"] != ""].iterrows():
            fig_scatter.add_annotation(
                x=row["Average"], y=row["Strike Rate"],
                text=row["Player"].split()[-1],
                showarrow=False, font=dict(size=9, color="#F5A623"), yshift=10,
            )
        style_chart(fig_scatter, "Strike Rate vs Batting Average (bubble = total runs)")
        st.plotly_chart(fig_scatter, width="stretch")
        insight_box(
            "Top-right quadrant (SR &gt; 130, Avg &gt; 30) represents elite all-format batters "
            "— high scoring rate <em>and</em> consistency."
        )

        # Top 10 run scorers
        st.markdown("#### Top 10 Run Scorers")
        top_bat = bat.nlargest(10, "runs")[["Player", "runs", "Strike Rate", "Average", "innings"]]
        fig_top = px.bar(
            top_bat, x="runs", y="Player",
            orientation="h",
            color="Strike Rate",
            color_continuous_scale="Oranges",
            hover_data={"Average": True, "innings": True},
        )
        fig_top.update_layout(yaxis=dict(categoryorder="total ascending"))
        style_chart(fig_top, "Top 10 IPL Run Scorers")
        st.plotly_chart(fig_top, width="stretch")

        divider()

        # Individual drilldown
        st.subheader("🔍 Individual Batter Drilldown")
        all_batters = sorted(bat["Player"].tolist())
        default_player = top10[0] if top10 else all_batters[0]
        selected_player = st.selectbox(
            "Select Batter", all_batters,
            index=all_batters.index(default_player),
            key="pi_batter",
        )

        col1, col2 = st.columns(2)

        phase_runs = _player_phase_runs(filtered, selected_player)
        fig_phase = px.bar(phase_runs, x="Phase", y="Runs",
                           color_discrete_sequence=["#F5A623"])
        style_chart(fig_phase, f"{selected_player} — Runs by Phase")
        col1.plotly_chart(fig_phase, width="stretch")

        inns_df = _innings_scores(filtered, selected_player)
        fig_cons = px.bar(inns_df, x="Innings #", y="Runs",
                          color="Runs", color_continuous_scale="Oranges")
        fig_cons.update_layout(coloraxis_showscale=False)
        style_chart(fig_cons, f"{selected_player} — Innings-by-Innings Scores")
        col2.plotly_chart(fig_cons, width="stretch")

        divider()

        # Bat position
        st.subheader("📍 Bat Position Analysis")
        bp = _bat_pos_stats(filtered)
        fig_bp = px.bar(
            bp, x="Bat Position", y="Strike Rate",
            color="Strike Rate", color_continuous_scale="Oranges",
        )
        fig_bp.update_layout(coloraxis_showscale=False)
        style_chart(fig_bp, "Strike Rate by Batting Position")
        st.plotly_chart(fig_bp, width="stretch")
        insight_box(
            "Lower-order batters (positions 7–11) often post higher strike rates "
            "due to short, explosive innings in the death overs."
        )

    # ═══════════════════════════════════════════════════════════════════════
    #  BOWLING
    # ═══════════════════════════════════════════════════════════════════════
    else:
        st.subheader("🎳 Bowling Analysis")
        with st.spinner("Loading insights…"):
            bowl = _bowling_stats(filtered, min_innings)

        if bowl.empty:
            st.warning("No data for the selected filters.")
            return

        # Economy vs Wickets scatter
        st.markdown("#### Economy vs Wickets Taken")
        top10_wkt = bowl.nlargest(10, "wickets")["Player"].tolist()
        bowl["Label"] = bowl["Player"].apply(lambda p: p if p in top10_wkt else "")

        fig_bscatter = px.scatter(
            bowl,
            x="wickets", y="Economy",
            size="balls", color="Bowling Avg",
            hover_name="Player",
            hover_data={"matches": True, "Bowling SR": True},
            color_continuous_scale="Blues_r",
        )
        fig_bscatter.add_hline(y=8.0, line_dash="dash", line_color="#444", opacity=0.7)
        for _, row in bowl[bowl["Label"] != ""].iterrows():
            fig_bscatter.add_annotation(
                x=row["wickets"], y=row["Economy"],
                text=row["Player"].split()[-1],
                showarrow=False, font=dict(size=9, color="#F5A623"), yshift=10,
            )
        style_chart(fig_bscatter, "Economy Rate vs Wickets (bubble = balls bowled)")
        st.plotly_chart(fig_bscatter, width="stretch")
        insight_box(
            "The ideal bowler sits bottom-right: <strong>high wickets, low economy</strong>. "
            "Bubble size represents total balls bowled."
        )

        # Top 10 wicket takers
        st.markdown("#### Top 10 Wicket Takers")
        top_bowl = bowl.nlargest(10, "wickets")[
            ["Player", "wickets", "Economy", "Bowling Avg", "matches"]
        ]
        fig_twkt = px.bar(
            top_bowl, x="wickets", y="Player",
            orientation="h",
            color="Economy", color_continuous_scale="Blues_r",
        )
        fig_twkt.update_layout(yaxis=dict(categoryorder="total ascending"))
        style_chart(fig_twkt, "Top 10 IPL Wicket Takers")
        st.plotly_chart(fig_twkt, width="stretch")

        divider()

        # Wicket type breakdown
        st.subheader("💥 Wicket Type Breakdown")
        wkt_types = (
            df[(df["striker_out"] == 1) & (df["wicket_kind"].notna())]
            ["wicket_kind"].value_counts().reset_index()
        )
        wkt_types.columns = ["Wicket Kind", "Count"]
        fig_wkt = px.pie(
            wkt_types,
            names="Wicket Kind", values="Count",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        style_chart(fig_wkt, "Distribution of Dismissal Types")
        st.plotly_chart(fig_wkt, width="stretch")

        divider()

        # Death over specialists
        st.subheader("☠️ Death Over Specialists (Overs 15–19)")
        death = filtered[(filtered["over"] >= 15) & (filtered["valid_ball"] == 1)].copy()
        death_agg = (
            death.groupby("bowler")
            .agg(runs=("runs_total", "sum"), balls=("valid_ball", "count"))
            .reset_index()
        )
        death_agg["Economy"] = (death_agg["runs"] / (death_agg["balls"] / 6)).round(2)
        death_agg = death_agg[death_agg["balls"] >= 60].sort_values("Economy").head(10)

        fig_death = px.bar(
            death_agg, x="Economy", y="bowler",
            orientation="h",
            color="Economy", color_continuous_scale="Blues_r",
        )
        fig_death.update_layout(
            yaxis=dict(categoryorder="total descending"),
            coloraxis_showscale=False,
        )
        style_chart(fig_death, "Best Death Over Bowlers (min 60 balls — lower is better)")
        st.plotly_chart(fig_death, width="stretch")
        insight_box(
            "Economy below <strong>8.5 in overs 15–19</strong> is the gold standard "
            "for death bowling specialists in the IPL."
        )
