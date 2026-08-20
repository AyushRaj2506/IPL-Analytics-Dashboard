import pandas as pd
import streamlit as st

TEAM_NORMALIZE = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
}


@st.cache_data
def load_data():
    """Load, normalize, and feature-engineer the ball-by-ball IPL dataset."""
    df = pd.read_csv(
        "data/IPL_data_ball_by_ball_Original_.csv", low_memory=False
    )

    # Normalize team names
    for col in ["batting_team", "bowling_team", "match_won_by", "toss_winner"]:
        if col in df.columns:
            df[col] = df[col].replace(TEAM_NORMALIZE)

    # Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Derived binary columns
    df["is_boundary"] = df["runs_batter"].isin([4, 6])
    df["is_six"] = df["runs_batter"] == 6
    df["is_four"] = df["runs_batter"] == 4
    df["is_dot"] = (df["runs_total"] == 0) & (df["valid_ball"] == 1)
    df["is_wicket"] = df["striker_out"] == 1

    # Phase labelling
    df["phase"] = pd.cut(
        df["over"],
        bins=[-1, 5, 14, 19],
        labels=["Powerplay (0-5)", "Middle (6-14)", "Death (15-19)"],
    )

    return df
