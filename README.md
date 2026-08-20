# 🏏 IPL Analytics Dashboard (2008–2026)

A professional sports analytics dashboard built with Python and Streamlit,
analyzing **295,000+ ball-by-ball records** across **1,243 IPL matches** from 2008 to 2026.

---

## 🔗 Live App
> Deploy to [Streamlit Community Cloud](https://streamlit.io/cloud) — add your link here after deployment.

---

## 📌 Key Insights
*(Fill in after running the app on your data)*

- Teams choosing to **field first** after winning the toss win a higher percentage of matches
- **Death-over economy below 8.5** correlates strongly with match wins
- **Required run rate** and **wickets remaining** are the two strongest win-probability predictors
- The win predictor achieves **~75–80% accuracy** on held-out 2nd-innings match states

---

## 🗺️ Pages

| Page | Description |
|---|---|
| 📊 Season Overview | KPIs, runs per season, toss impact, top venues |
| 🏆 Team Analysis | H2H, win % trend, phase batting/bowling, venue heatmap |
| 🎯 Player Intelligence | SR vs Avg scatter, consistency, bowling analysis, bat position |
| 📈 Phase Analysis | Powerplay deep dive, death specialists, chase/defend, DRS, superovers |
| 🤖 Win Predictor | Live XGBoost-powered win probability with gauge charts |

---

## 🛠 Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3.11+** | Core language |
| **Pandas / NumPy** | Data processing (295K rows) |
| **XGBoost** | Win probability ML model |
| **Scikit-learn** | Train/test split, metrics |
| **Plotly** | All interactive charts |
| **Streamlit** | Web application framework |
| **Joblib** | Model persistence |

---

## 📁 Dataset

Ball-by-ball IPL data (2008–2026):
- **295,732 records** across **64 columns**
- **1,243 matches** · **19 teams** · **All venues**
- Source file: `data/IPL_data_ball_by_ball_Original_.csv`

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd ipl-dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place dataset
# Put IPL_data_ball_by_ball_Original_.csv in the data/ folder

# 4. Run the app
streamlit run app.py
```

The ML model (`modules/win_model.pkl`) is auto-trained on first run and cached for subsequent runs.

---

## 📂 Project Structure

```
ipl-dashboard/
├── app.py                    # Main Streamlit entry point
├── data/
│   └── IPL_data_ball_by_ball_Original_.csv
├── modules/
│   ├── __init__.py
│   ├── data_loader.py        # Cached data load + normalization
│   ├── overview.py           # Page 1: Season Overview
│   ├── team_analysis.py      # Page 2: Team Analysis
│   ├── player_intel.py       # Page 3: Player Intelligence
│   ├── phase_analysis.py     # Page 4: Phase & Situational Analysis
│   └── win_predictor.py      # Page 5: Win Probability Predictor
├── .streamlit/
│   └── config.toml           # Dark theme
└── requirements.txt
```

---

## ✨ Features

- **Dark-themed** premium UI with custom CSS metric cards
- **Fully interactive** Plotly charts with hover tooltips
- **`@st.cache_data`** on all heavy computations for fast navigation
- **Sidebar season filter** persisting across pages
- **XGBoost win predictor** with live gauge charts and feature importance
- Mobile-responsive layouts using `st.columns`
