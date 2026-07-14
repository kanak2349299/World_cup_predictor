import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl

st.set_page_config(page_title="FIFA World Cup Match Predictor", page_icon="⚽", layout="wide")

@st.cache_resource
def load_artifacts():
    model = pkl.load(open("world_cup_predictor.pkl", "rb"))
    winner_encoder = pkl.load(open("winner_encoder.pkl", "rb"))
    feature_cols = pkl.load(open("feature_columns.pkl", "rb"))

    problems = []
    if not hasattr(model, "predict"):
        problems.append("`world_cup_predictor.pkl` does not look like a trained model (no `.predict`).")
    if not (hasattr(winner_encoder, "classes_") and hasattr(winner_encoder, "transform")):
        problems.append("`winner_encoder.pkl` does not look like a LabelEncoder (no `.classes_` / `.transform`).")
    if not isinstance(feature_cols, (list, tuple)):
        problems.append("`feature_columns.pkl` should be the list of feature column names saved by train_model.py.")

    if problems:
        st.error(
            "⚠️ One or more saved files don't match what the app expects:\n\n"
            + "\n".join(f"- {p}" for p in problems)
            + "\n\nRun `train_model.py` again and make sure all three files are "
              "saved fresh, then restart the app."
        )
        st.stop()

    return model, winner_encoder, list(feature_cols)

model, winner_encoder, feature_cols = load_artifacts()

STAGE_COLUMNS = [c for c in feature_cols if c.startswith("stage_")]
STAGE_NAMES = [c[len("stage_"):] for c in STAGE_COLUMNS]

TEAM_NAMES = sorted([
    "Argentina", "Brazil", "France", "Germany", "Italy", "Spain", "England",
    "Netherlands", "Portugal", "Belgium", "Uruguay", "Croatia", "Morocco",
    "Switzerland", "USA", "Mexico", "Japan", "South Korea", "Senegal",
    "Ghana", "Nigeria", "Cameroon", "Tunisia", "Egypt", "Algeria",
    "Poland", "Denmark", "Sweden", "Serbia", "Wales", "Australia",
    "Canada", "Costa Rica", "Ecuador", "Iran", "Saudi Arabia", "Qatar",
])

if "team_history" not in st.session_state:
    st.session_state["team_history"] = {
        t: {"played": 0, "wins": 0, "goals_for": 0, "goals_against": 0} for t in TEAM_NAMES
    }

if "h2h_history" not in st.session_state:
    st.session_state["h2h_history"] = {}

def get_team_features(team):
    h = st.session_state["team_history"].get(team, {"played": 0, "wins": 0, "goals_for": 0, "goals_against": 0})
    winrate = h["wins"] / h["played"] if h["played"] > 0 else 0.5
    goal_diff_avg = (h["goals_for"] - h["goals_against"]) / h["played"] if h["played"] > 0 else 0
    return winrate, goal_diff_avg, h["played"]

def get_h2h_feature(home_team, away_team):
    key = tuple(sorted([home_team, away_team]))
    s = st.session_state["h2h_history"].get(key, {"home_wins": 0, "away_wins": 0, "draws": 0})
    total = s["home_wins"] + s["away_wins"] + s["draws"]
    return s["home_wins"] / total if total > 0 else 0.5

st.markdown("""
<style>
[data-testid="stAppViewContainer"]{
    background: linear-gradient(-45deg, #012a1c, #014421, #0a5c36, #013220, #0b3d2e);
    background-size: 400% 400%;
    animation: gradientShift 12s ease infinite;
    position: relative;
}
[data-testid="stHeader"]{ background: transparent; z-index: 999; }
html, body{ overflow-x: hidden; }
[data-testid="collapsedControl"]{ z-index: 999; }
[data-testid="stSidebar"]{
    z-index: 100;
    transition: margin-left 0.3s ease-in-out, width 0.3s ease-in-out;
}
@keyframes gradientShift{
    0%{background-position:0% 50%;}
    50%{background-position:100% 50%;}
    100%{background-position:0% 50%;}
}
.football{
    position: fixed; font-size: 40px; opacity: 0.12; z-index: 0;
    pointer-events: none; animation-name: floatBall;
    animation-timing-function: ease-in-out; animation-iteration-count: infinite;
}
@keyframes floatBall{
    0%{ transform: translateY(0) translateX(0) rotate(0deg); }
    25%{ transform: translateY(-40px) translateX(20px) rotate(90deg); }
    50%{ transform: translateY(-10px) translateX(-15px) rotate(180deg); }
    75%{ transform: translateY(30px) translateX(25px) rotate(270deg); }
    100%{ transform: translateY(0) translateX(0) rotate(360deg); }
}
.b1{ top:8%; left:5%; animation-duration: 14s; }
.b2{ top:20%; left:80%; animation-duration: 18s; }
.b3{ top:55%; left:15%; animation-duration: 16s; }
.b4{ top:70%; left:70%; animation-duration: 20s; }
.b5{ top:35%; left:45%; animation-duration: 12s; }
.b6{ top:85%; left:30%; animation-duration: 17s; }
.b7{ top:10%; left:55%; animation-duration: 15s; }

[data-testid="stMain"].block-container{
    background: rgba(0,10,5,0.6); backdrop-filter: blur(6px);
    border-radius: 18px; padding: 2rem 2.5rem;
    box-shadow: 0 0 25px rgba(0,0,0,0.4); position: relative; z-index: 1;
}
h1, h2, h3, h4, p, span, label, li{ color:white!important; }
h1{
    text-align:center; text-shadow:0 0 12px #00ff66, 0 0 4px black;
    animation:glow 2s infinite alternate;
}
@keyframes glow{ from{ text-shadow:0 0 10px #00ff66; } to{ text-shadow:0 0 30px gold; } }
.stButton>button{
    background:#00C853; color:white; border-radius:30px; font-size:18px;
    font-weight:bold; padding:12px 30px; transition:0.4s; border:none;
}
.stButton>button:hover{
    transform:scale(1.08); background:#FFD700; color:black;
    box-shadow:0px 0px 20px gold;
}
[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#001b2e 0%,#013220 55%,#00251a 100%);
    box-shadow: 4px 0px 25px rgba(0,255,120,0.15);
    border-right: 1px solid rgba(0,255,150,0.15);
}
[data-testid="stSidebar"] *{ color:white!important; }
.dash-title{
    text-align:center; font-size:26px; font-weight:800; letter-spacing:1px;
    margin: 6px 0 2px 0;
    background: linear-gradient(90deg, #00ff87, #ffd700, #00ff87);
    background-size: 200% auto; -webkit-background-clip: text;
    -webkit-text-fill-color: transparent; animation: shine 3s linear infinite;
}
@keyframes shine{ 0%{ background-position: 0% center; } 100%{ background-position: 200% center; } }
.dash-subtitle{
    text-align:center; font-size:12px; letter-spacing:3px; text-transform:uppercase;
    color:#8effc1!important; opacity:0.85; margin-bottom:10px;
}
.pitch-divider{
    height:3px; margin:14px 0; border-radius:3px;
    background: linear-gradient(90deg, transparent, #00ff66, gold, #00ff66, transparent);
    background-size: 200% auto; animation: shine 4s linear infinite;
}
.dash-card{
    background: rgba(255,255,255,0.06); border: 1px solid rgba(0,255,150,0.25);
    border-radius: 14px; padding: 14px 16px; margin-bottom: 14px; transition: all 0.3s ease;
}
.dash-card:hover{
    background: rgba(255,255,255,0.11); border-color: gold;
    transform: translateY(-2px); box-shadow: 0 0 18px rgba(0,255,120,0.25);
}
.dash-card h4{ margin:0 0 8px 0!important; font-size:15px!important; color:#ffd700!important; letter-spacing:0.5px; }
.dash-card p,.dash-card li{ font-size:13px!important; line-height:1.5; margin:0; }
.pill-row{ display:flex; flex-wrap:wrap; gap:6px; margin-top:6px; }
.pill{
    background: rgba(0,255,120,0.12); border:1px solid rgba(0,255,120,0.35);
    border-radius:20px; padding:4px 10px; font-size:12px!important; white-space:nowrap;
}
[data-testid="stSidebar"].stSelectbox label{ font-weight:700!important; color:#8effc1!important; }
[data-testid="stSidebar"].stAlert{ background: rgba(255,255,255,0.08)!important; border-radius: 10px; }
.dev-card{
    text-align:center; background: rgba(255,255,255,0.06);
    border:1px solid rgba(255,215,0,0.35); border-radius:14px; padding:14px;
}
.dev-card img{ border-radius:50%; }
</style>

<div class="football b1">⚽</div>
<div class="football b2">⚽</div>
<div class="football b3">⚽</div>
<div class="football b4">⚽</div>
<div class="football b5">⚽</div>
<div class="football b6">⚽</div>
<div class="football b7">⚽</div>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 style='font-size:55px;'>🏆FIFA World Cup Match Predictor</h1>",
    unsafe_allow_html=True,
)

def safe_image(url, fallback_emoji, use_container_width=True):
    try:
        st.image(url, use_container_width=use_container_width)
    except Exception:
        st.markdown(
            f"<div style='text-align:center;font-size:80px;'>{fallback_emoji}</div>",
            unsafe_allow_html=True,
        )

with st.sidebar:
    st.markdown('<div class="dash-title">✨DASHBOARD</div>', unsafe_allow_html=True)
    st.markdown('<div class="dash-subtitle">Predict · Analyze · Win</div>', unsafe_allow_html=True)
    st.markdown('<div class="pitch-divider"></div>', unsafe_allow_html=True)

    page = st.selectbox(
        "📌 Navigate",
        (
            "🏠 Home",
            "🏅 Match Prediction",
            "📊 Dataset Information",
            "🤖 Model Information",
            "🛠️ Tech Stack",
            "📈 Project Insights",
            "👩🏻‍💻 Developer",
        ),
    )
    st.markdown('<div class="pitch-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="dash-card">
        <h4>📊 Dataset</h4>
        <p><b>Competition:</b> FIFA World Cup</p>
        <p><b>Years Covered:</b> 1974 – 2022</p>
        <p><b>Records:</b> Historical match data</p>
        <p><b>Target Variable:</b> Winner</p>
        <div class="pill-row">
            <span class="pill">🌍 Year</span>
            <span class="pill">🏟 Stage</span>
            <span class="pill">🏠 Home Team</span>
            <span class="pill">✈ Away Team</span>
            <span class="pill">📈 Win Rate</span>
            <span class="pill">🤝 Head-to-Head</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="dash-card">
        <h4>🤖 Machine Learning</h4>
        <p>✅ {type(model).__name__}</p>
        <p style="margin-top:8px;opacity:0.85;">Predicts the match winner using pre-match
        stats only — no live score input, so no leakage.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="dash-card">
        <h4>🛠 Tech Stack</h4>
        <div class="pill-row">
            <span class="pill">🐍 Python</span>
            <span class="pill">📊 Pandas</span>
            <span class="pill">🔢 NumPy</span>
            <span class="pill">🤖 Scikit-learn</span>
            <span class="pill">🌐 Streamlit</span>
            <span class="pill">💾 Pickle</span>
            <span class="pill">📈 Matplotlib</span>
            <span class="pill">📋 Jupyter</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="dash-card">
        <h4>⚽ Features Used</h4>
        <div class="pill-row">
            <span class="pill">✔ Home/Away Team</span>
            <span class="pill">✔ Match Stage</span>
            <span class="pill">✔ Win Rate</span>
            <span class="pill">✔ Goal Diff Avg</span>
            <span class="pill">✔ Head-to-Head</span>
            <span class="pill">✔ Tournament Year</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="dev-card">
        <h4 style="color:#ffd700;margin-bottom:6px;">💁🏻‍♀️ Developer</h4>
        <p style="font-size:15px;font-weight:700;margin-bottom:4px;">Himangi Gupta</p>
        <p style="font-size:12px;opacity:0.8;">🔗 <a href="https://github.com/kanak2349299" style="color:#8effc1!important;">GitHub Profile</a></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="pitch-divider"></div>', unsafe_allow_html=True)
    safe_image(
        "https://cdn.pixabay.com/photo/2015/10/16/12/25/trophy-984958_960_720.png",
        "🏆",
    )
    st.markdown(
        "<p style='text-align:center;font-size:12px;opacity:0.75;'>🏅 Turning Football Data into Predictions 😊</p>",
        unsafe_allow_html=True,
    )

def render_home():
    st.subheader("Predict the outcome of FIFA World Cup matches!")
    st.markdown(f"""
    <div class="dash-card">
        <h4>👋 Welcome</h4>
        <p>This app uses a {type(model).__name__} model trained on historical FIFA World Cup
        matches from 1974 to 2022 to predict a match winner.</p>
        <p>Use the <b>📌 Navigate</b> menu on the left to explore the dataset, the model,
        the tech stack behind this project, or jump straight into
        <b>🏅 Match Prediction</b> to try it yourself.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="dash-card" style="text-align:center;">
            <h4>📅 1974–2022</h4>
            <p>Years of World Cup data</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="dash-card" style="text-align:center;">
            <h4>🌳 {type(model).__name__}</h4>
            <p>Classifier model</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="dash-card" style="text-align:center;">
            <h4>🎯 Pre-Match</h4>
            <p>No score leakage</p>
        </div>""", unsafe_allow_html=True)

def render_prediction():
    st.header("🏅Match Winner Prediction")
    st.caption("Prediction is made **before** the match — no score input needed. "
               "It's based on each team's win rate, goal difference, and head-to-head history.")

    year = st.number_input("🌍Tournament Year", min_value=1974, max_value=2040, value=2026, step=4)
    stage_display = st.selectbox("🏆Match Stage", STAGE_NAMES) if STAGE_NAMES else None
    home_team = st.selectbox("🏠Home Team", TEAM_NAMES, index=TEAM_NAMES.index("Argentina") if "Argentina" in TEAM_NAMES else 0)
    away_team = st.selectbox("✈Away Team", [t for t in TEAM_NAMES if t!= home_team])

    if st.button("🏆 Predict Winner"):
        home_winrate, home_gd, home_played = get_team_features(home_team)
        away_winrate, away_gd, away_played = get_team_features(away_team)
        h2h_home_rate = get_h2h_feature(home_team, away_team)

        row = {c: 0 for c in feature_cols}
        row["year"] = year
        row["home_winrate"] = home_winrate
        row["away_winrate"] = away_winrate
        row["home_goal_diff_avg"] = home_gd
        row["away_goal_diff_avg"] = away_gd
        row["home_matches_played"] = home_played
        row["away_matches_played"] = away_played
        row["h2h_home_win_rate"] = h2h_home_rate
        if stage_display is not None:
            stage_col = f"stage_{stage_display}"
            if stage_col in row:
                row[stage_col] = 1

        input_df = pd.DataFrame([row], columns=feature_cols)
        probs = model.predict_proba(input_df)[0]
        classes = list(winner_encoder.classes_)
        home_prob = probs[classes.index(home_team)]
        away_prob = probs[classes.index(away_team)]
        winner = home_team if home_prob >= away_prob else away_team 

        st.session_state["winner"] = winner
        st.success(f"🏆 Predicted Winner: **{winner}** 🎉")
        st.balloons()

    if "winner" in st.session_state:
        st.markdown(f"""
        <div style="
            padding:25px; border-radius:20px; font-size:30px;
            background:linear-gradient(135deg,#004d40,#1b5e20);
            color:white; text-align:center; animation: fade 2s;">🏆 Predicted Winner<br><br>
            <b>{st.session_state['winner']}</b>
        </div>
        <style>
        @keyframes fade{{ from{{ opacity:0; transform:translateY(60px); }} to{{ opacity:1; transform:translateY(0px); }} }}
        </style>
        """, unsafe_allow_html=True)

def render_dataset():
    st.header("📊 Dataset Information")
    st.markdown("""
    <div class="dash-card">
        <h4>📊 Dataset</h4>
        <p><b>Competition:</b> FIFA World Cup</p>
        <p><b>Years Covered:</b> 1974 – 2022</p>
        <p><b>Records:</b> Historical FIFA World Cup matches</p>
        <p><b>Target Variable:</b> Winner</p>
        <div class="pill-row">
            <span class="pill">🌍 Year</span>
            <span class="pill">🏟 Stage</span>
            <span class="pill">🏠 Home Team</span>
            <span class="pill">✈ Away Team</span>
            <span class="pill">📈 Win Rate</span>
            <span class="pill">📊 Goal Difference</span>
            <span class="pill">🤝 Head-to-Head</span>
            <span class="pill">🏆 Winner</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="dash-card">
        <h4>🏟 Available Match Stages</h4>
        <div class="pill-row">
    """ + "".join(f'<span class="pill">{s}</span>' for s in STAGE_NAMES) + """
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="dash-card">
        <h4>🌍 Countries Available</h4>
        <div class="pill-row">
    """ + "".join(f'<span class="pill">{t}</span>' for t in TEAM_NAMES) + """
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_model_info():
    st.header("🤖 Model Information")
    model_name = type(model).__name__
    st.markdown(f"""
    <div class="dash-card">
        <h4>🤖 Machine Learning</h4>
        <p>✅ {model_name}</p>
        <p style="margin-top:8px;opacity:0.85;">Predicts the winner of a FIFA World Cup match
        <b>before it's played</b>, using each team's win rate, average goal difference,
        head-to-head history, tournament year, and match stage — no live score is used
        as input, so this reflects genuine pre-match predictive skill rather than reading
        off the final result.</p>
    </div>
    """, unsafe_allow_html=True)
    st.info("Run `train_model.py` to see this run's exact cross-validated and test "
            "accuracy printed in the console — it will vary depending on your data split "
            "and hyperparameters.")

def render_tech_stack():
    st.header("🛠️ Tech Stack")
    st.markdown("""
    <div class="dash-card">
        <h4>🛠 Tools Used</h4>
        <div class="pill-row">
            <span class="pill">🐍 Python</span>
            <span class="pill">📊 Pandas</span>
            <span class="pill">🔢 NumPy</span>
            <span class="pill">🤖 Scikit-learn</span>
            <span class="pill">🌐 Streamlit</span>
            <span class="pill">💾 Pickle</span>
            <span class="pill">📈 Matplotlib</span>
            <span class="pill">📋 Jupyter Notebook</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_insights():
    st.header("📈 Project Insights")
    st.markdown("""
    <div class="dash-card">
        <h4>⚽ Features Used</h4>
        <div class="pill-row">
            <span class="pill">✔ Home Team</span>
            <span class="pill">✔ Away Team</span>
            <span class="pill">✔ Match Stage</span>
            <span class="pill">✔ Win Rate</span>
            <span class="pill">✔ Goal Difference Avg</span>
            <span class="pill">✔ Head-to-Head Record</span>
            <span class="pill">✔ Tournament Year</span>
        </div>
    </div>
    <div class="dash-card">
        <h4>💡 Key Takeaways</h4>
        <p>• This model predicts <b>before</b> the match is played — it never sees the
        score, only each team's win rate, goal difference, and head-to-head history
        going into the match.</p>
        <p>• Match stage is one-hot encoded and team win-rate/goal-diff are computed
        from running history, rather than treating team names as arbitrary IDs.</p>
        <p>• Predictions naturally improve as more matches are played in a session,
        since win-rate and head-to-head stats update after every prediction.</p>
    </div>
    """, unsafe_allow_html=True)

def render_developer():
    st.header("👩🏻‍💻 Developer")
    st.markdown("""
    <div class="dev-card">
        <p style="font-size:15px;font-weight:700;margin-bottom:4px;">Himangi Gupta</p>
        <p style="font-size:12px;opacity:0.8;">🔗 <a href="https://github.com/kanak2349299" style="color:#8effc1!important;">GitHub Profile</a></p>
    </div>
    """, unsafe_allow_html=True)

PAGES = {
    "🏠 Home": render_home,
    "🏅 Match Prediction": render_prediction,
    "📊 Dataset Information": render_dataset,
    "🤖 Model Information": render_model_info,
    "🛠️ Tech Stack": render_tech_stack,
    "📈 Project Insights": render_insights,
    "👩🏻‍💻 Developer": render_developer,
}

PAGES[page]()
