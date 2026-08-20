"""
Page 5 — Win Probability Predictor
XGBoost model trained on 2nd-innings match states. Outputs live win probability
via a Plotly gauge chart based on user-controlled sliders.
"""

import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from modules.ui_helpers import divider, insight_box, kpi_card, page_header, style_chart


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_PATH = "modules/win_model.pkl"
FEATURES = [
    "runs_needed",
    "balls_remaining",
    "wickets_remaining",
    "current_rr",
    "required_rr",
    "target",
]
MAX_OVERS = 20
MIN_SAMPLES = 200   # minimum rows needed to train


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

@st.cache_data
def _prepare_chase_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the training dataset from 2nd-innings rows.
    Robustly handles:
      - innings column being numeric (2) or string ('2')
      - missing runs_target values
      - empty result after filters
    """
    # Support both numeric and string innings values
    innings_vals = df["innings"].dropna().unique()
    if len(innings_vals) == 0:
        return pd.DataFrame()

    # Try numeric 2 first, fall back to string '2'
    chase = df[df["innings"] == 2].copy()
    if chase.empty:
        chase = df[df["innings"] == "2"].copy()
    if chase.empty:
        chase = df[df["innings"].astype(str).str.strip() == "2"].copy()

    # Guard: must have runs_target
    chase = chase[chase["runs_target"].notna()]
    if chase.empty:
        return pd.DataFrame()

    chase["runs_needed"] = (chase["runs_target"] - chase["team_runs"]).clip(lower=0)
    chase["balls_remaining"] = (MAX_OVERS * 6 - chase["team_balls"]).clip(lower=0)
    chase["wickets_remaining"] = (10 - chase["team_wicket"]).clip(lower=0, upper=10)

    # Current run rate (avoid div/0)
    safe_balls = chase["team_balls"].replace(0, np.nan)
    chase["current_rr"] = (
        (chase["team_runs"] / (safe_balls / 6)).fillna(0).clip(upper=36)
    )

    # Required run rate
    safe_br = chase["balls_remaining"].replace(0, np.nan)
    chase["required_rr"] = (
        (chase["runs_needed"] / (safe_br / 6))
        .replace([float("inf"), -float("inf")], 36)
        .fillna(36)
        .clip(upper=36)
    )

    chase["target"] = chase["runs_target"]
    chase["batting_team_won"] = (
        chase["batting_team"] == chase["match_won_by"]
    ).astype(int)

    # Use end-of-over snapshots when ball_no is present
    if "ball_no" in chase.columns:
        snapshot = chase[chase["ball_no"] == 6]
        if len(snapshot) >= MIN_SAMPLES:
            chase = snapshot

    result = chase[FEATURES + ["batting_team_won"]].dropna()
    return result


# ---------------------------------------------------------------------------
# Model training / loading
# ---------------------------------------------------------------------------

def _train_model(chase_df: pd.DataFrame):
    X = chase_df[FEATURES]
    y = chase_df["batting_team_won"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        random_state=42, eval_metric="logloss", verbosity=0,
    )
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    joblib.dump(model, MODEL_PATH)
    return model, acc, auc


def _load_or_train(chase_df: pd.DataFrame):
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            sample = chase_df[FEATURES].iloc[:1]
            _ = model.predict_proba(sample)   # quick validity check
            X = chase_df[FEATURES]
            y = chase_df["batting_team_won"]
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            acc = accuracy_score(y_test, model.predict(X_test))
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            return model, acc, auc
        except Exception:
            pass   # fall through → retrain
    return _train_model(chase_df)


# ---------------------------------------------------------------------------
# Gauge chart
# ---------------------------------------------------------------------------

def _gauge(prob: float, title: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 46, "color": "white",
                                        "family": "Inter, sans-serif"}},
        title={"text": title, "font": {"size": 15, "color": "#aaa",
                                       "family": "Inter, sans-serif"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#555",
                     "tickfont": {"color": "#aaa"}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0, 30],  "color": "rgba(60,20,20,0.4)"},
                {"range": [30, 70], "color": "rgba(20,30,60,0.4)"},
                {"range": [70, 100],"color": "rgba(20,60,20,0.4)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.85,
                "value": round(prob * 100, 1),
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white", "family": "Inter, sans-serif"},
        height=300,
        margin=dict(l=30, r=30, t=60, b=10),
    )
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def show(df: pd.DataFrame):
    page_header(
        "🤖", "Win Probability Predictor",
        "An XGBoost model trained on 2nd-innings match states — "
        "adjust the sliders to simulate any chase scenario live."
    )

    # ── Prepare & load/train ──────────────────────────────────────────────────
    with st.spinner("Preparing chase data and loading ML model…"):
        chase_df = _prepare_chase_df(df)

    # Guard: insufficient data
    if len(chase_df) < MIN_SAMPLES:
        st.error(
            f"⚠️ Not enough 2nd-innings rows with a known target to train the model "
            f"(found {len(chase_df)}, need ≥ {MIN_SAMPLES}). "
            f"Check that `innings`, `runs_target`, `team_runs`, and `team_balls` "
            f"columns are populated in the dataset."
        )
        st.info(
            "**Debug info:** "
            f"innings unique values = `{df['innings'].dropna().unique()[:10].tolist()}` | "
            f"rows with innings==2: `{(df['innings']==2).sum()}` | "
            f"rows with runs_target not-null: `{df['runs_target'].notna().sum()}`"
        )
        return

    with st.spinner("Training / loading XGBoost model…"):
        model, acc, auc = _load_or_train(chase_df)

    # Success banner
    st.markdown(
        f"""
        <div style="background:#0e2818; border-left:4px solid #2ecc71;
                    border-radius:8px; padding:14px 20px; margin:0 0 16px 0;
                    font-family:'Inter',sans-serif">
          <span style="color:#2ecc71; font-weight:700">✅ Model Ready&nbsp;&nbsp;</span>
          <span style="color:#ccc">
            Accuracy: <strong style="color:white">{acc*100:.1f}%</strong> &nbsp;·&nbsp;
            AUC-ROC: <strong style="color:white">{auc:.3f}</strong> &nbsp;·&nbsp;
            Trained on <strong style="color:white">{len(chase_df):,}</strong> ball snapshots
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "XGBoost · 200 estimators · max_depth=5 · "
        "2nd-innings end-of-over snapshots from IPL 2008–2026"
    )

    divider()

    # ── Sliders ───────────────────────────────────────────────────────────────
    st.subheader("🎛️ Match State Sliders")
    st.markdown(
        "<p style='color:#888;font-size:0.9rem;margin:-8px 0 16px 0'>"
        "Simulate any chase scenario and see live win probabilities.</p>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        target       = st.slider("🎯 Target (Runs)",        50, 300, 165, 1, key="wp_target")
        overs_done   = st.slider("⏱️ Overs Completed",       0,  19,  10, 1, key="wp_overs")
        wickets_lost = st.slider("💥 Wickets Lost",          0,   9,   3, 1, key="wp_wkts")
    with col2:
        runs_so_far  = st.slider("🏏 Runs Scored So Far",   0, target - 1, min(80, target - 1), 1, key="wp_runs")

    # Derived
    runs_needed       = max(target - runs_so_far, 0)
    balls_done        = overs_done * 6
    balls_remaining   = max(MAX_OVERS * 6 - balls_done, 1)
    wickets_remaining = 10 - wickets_lost
    current_rr  = round(runs_so_far / (balls_done / 6) if balls_done > 0 else 0.0, 2)
    required_rr = round(min(runs_needed / (balls_remaining / 6), 36.0), 2)

    divider()

    # Live derived KPIs
    k1, k2, k3, k4 = st.columns(4)
    kpi_card(k1, "Runs Needed",      f"{runs_needed}",       "to win")
    kpi_card(k2, "Balls Remaining",  f"{balls_remaining}",   f"{20 - overs_done} overs left")
    kpi_card(k3, "Current RR",       f"{current_rr:.2f}",    "runs/over so far")
    kpi_card(k4, "Required RR",      f"{required_rr:.2f}",   "runs/over needed")

    divider()
    st.subheader("📊 Win Probability")

    input_row = pd.DataFrame([{
        "runs_needed":       runs_needed,
        "balls_remaining":   balls_remaining,
        "wickets_remaining": wickets_remaining,
        "current_rr":        current_rr,
        "required_rr":       required_rr,
        "target":            target,
    }])
    proba      = model.predict_proba(input_row)[0]
    chase_prob = proba[1]
    defend_prob = proba[0]

    g1, g2 = st.columns(2)
    g1.plotly_chart(_gauge(chase_prob,  "🏏 Chasing Team",   "#F5A623"), width="stretch")
    g2.plotly_chart(_gauge(defend_prob, "🎳 Defending Team", "#1C6EF5"), width="stretch")

    # Live commentary
    if chase_prob > 0.70:
        commentary = (f"🟢 The chasing team is in a <strong>dominant position</strong> "
                      f"({chase_prob*100:.1f}% win probability).")
    elif chase_prob > 0.55:
        commentary = (f"🟡 A <strong>tight contest</strong>, slightly favouring the "
                      f"chasing team ({chase_prob*100:.1f}%).")
    elif chase_prob > 0.45:
        commentary = "⚖️ <strong>Neck-and-neck!</strong> Both sides have near-equal chances."
    elif chase_prob > 0.30:
        commentary = (f"🟠 The <strong>defending team</strong> has the upper hand "
                      f"({defend_prob*100:.1f}% win probability).")
    else:
        commentary = (f"🔴 The chasing team is under severe pressure — "
                      f"defending team at <strong>{defend_prob*100:.1f}%</strong>.")
    insight_box(commentary)

    divider()

    # ── Feature Importance ───────────────────────────────────────────────────
    st.subheader("📌 Feature Importance")
    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=True)

    fig_imp = go.Figure(go.Bar(
        x=importance["Importance"],
        y=importance["Feature"],
        orientation="h",
        marker=dict(color=importance["Importance"], colorscale="Oranges"),
    ))
    fig_imp.update_layout(xaxis_title="Importance")
    style_chart(fig_imp, "XGBoost Feature Importance (F-Score)")
    st.plotly_chart(fig_imp, width="stretch")
    insight_box(
        "<code>required_rr</code> and <code>wickets_remaining</code> are typically the "
        "strongest predictors — a steep required run rate with few wickets left makes "
        "a successful chase nearly impossible."
    )
