import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import streamlit.components.v1 as components

# -----------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Flight Satisfaction Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------
# Load model + label encoder (cached so it only loads once)
# -----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("airline_satisfaction_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, label_encoder

try:
    model, label_encoder = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Make sure `airline_satisfaction_model.pkl` "
        "and `label_encoder.pkl` are in the same folder as this app."
    )
    st.stop()

RATING_COLS = [
    "Seat comfort",
    "Departure/Arrival time convenient",
    "Food and drink",
    "Gate location",
    "Inflight wifi service",
    "Inflight entertainment",
    "Online support",
    "Ease of Online booking",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Cleanliness",
    "Online boarding",
]

# -----------------------------------------------------------------------
# Feature engineering (mirrors the notebook exactly)
# -----------------------------------------------------------------------
def delay_category(x):
    if x == 0:
        return "No Delay"
    elif x <= 15:
        return "Short Delay"
    elif x <= 60:
        return "Medium Delay"
    else:
        return "Long Delay"


def age_group(x):
    if x < 25:
        return "Young"
    elif x < 55:
        return "Adult"
    else:
        return "Senior"


def distance_category(x):
    if x < 1000:
        return "Short Distance"
    elif x < 2500:
        return "Medium Distance"
    else:
        return "Long Distance"


def build_feature_row(raw: dict) -> pd.DataFrame:
    row = dict(raw)

    total_delay = row["Departure Delay in Minutes"] + row["Arrival Delay in Minutes"]
    row["Total Delay"] = total_delay
    row["Delay Category"] = delay_category(total_delay)
    row["Service Score"] = np.mean([row[c] for c in RATING_COLS])
    row["Age Group"] = age_group(row["Age"])
    row["Flight Distance Category"] = distance_category(row["Flight Distance"])

    column_order = [
        "Customer Type", "Age", "Type of Travel", "Class", "Flight Distance",
        "Seat comfort", "Departure/Arrival time convenient", "Food and drink",
        "Gate location", "Inflight wifi service", "Inflight entertainment",
        "Online support", "Ease of Online booking", "On-board service",
        "Leg room service", "Baggage handling", "Checkin service",
        "Cleanliness", "Online boarding", "Departure Delay in Minutes",
        "Arrival Delay in Minutes", "Total Delay", "Delay Category",
        "Service Score", "Age Group", "Flight Distance Category",
    ]
    return pd.DataFrame([row])[column_order]


# =========================================================================
# THEME — "Departure Board" design system
# =========================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root {
    --navy-deep:   #0B1D33;
    --navy-panel:  #12304F;
    --navy-line:   #24476B;
    --amber:       #FFB300;
    --amber-soft:  #FFD569;
    --ink:         #0B1D33;
    --paper:       #F6F8FB;
    --slate:       #6C84A3;
    --green:       #2FBF71;
    --red:         #FF5C5C;
}

/* ---- Global chrome cleanup ---- */
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 1.2rem; max-width: 1100px;}
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--ink);
}

/* ---- Hero / departure board strip ---- */
.hero {
    background: linear-gradient(135deg, var(--navy-deep) 0%, #16385C 100%);
    border-radius: 14px;
    padding: 34px 38px 28px 38px;
    margin-bottom: 28px;
    box-shadow: 0 10px 30px rgba(11,29,51,0.25);
    animation: rise 0.7s cubic-bezier(.2,.8,.2,1) both;
}
.board-strip {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12.5px;
    letter-spacing: 1.5px;
    color: var(--amber-soft);
    display: flex;
    gap: 22px;
    flex-wrap: wrap;
    border-bottom: 1px dashed rgba(255,179,0,0.35);
    padding-bottom: 12px;
    margin-bottom: 16px;
}
.board-strip span.dot {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--green);
    margin-right: 6px;
    box-shadow: 0 0 6px var(--green);
}
.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    color: white;
    font-size: 30px;
    font-weight: 700;
    margin: 0 0 6px 0;
    letter-spacing: -0.3px;
}
.hero p {
    color: #A9C1DE;
    font-size: 15px;
    margin: 0;
    max-width: 640px;
    line-height: 1.55;
}

@keyframes rise {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ---- Section labels ---- */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    letter-spacing: 1px;
    color: var(--slate);
    margin: 26px 0 6px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #E1E7F0;
}

/* ---- Panel card wrapping the form ---- */
div[data-testid="stForm"] {
    background: white;
    border: 1px solid #E1E7F0;
    border-radius: 14px;
    padding: 26px 28px 14px 28px;
    box-shadow: 0 2px 10px rgba(11,29,51,0.04);
}

/* ---- Sliders use theme primaryColor (see .streamlit/config.toml) ---- */

/* ---- Submit button ---- */
.stButton>button, button[kind="primaryFormSubmit"], button[kind="formSubmit"] {
    background: var(--amber) !important;
    color: var(--navy-deep) !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1rem !important;
    letter-spacing: 0.3px;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton>button:hover, button[kind="primaryFormSubmit"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(255,179,0,0.35);
}

/* ---- Boarding pass result card ---- */
.pass-wrap {
    display: flex;
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 12px 28px rgba(11,29,51,0.14);
    animation: rise 0.5s ease both;
    margin-top: 6px;
}
.pass-main {
    flex: 3;
    background: var(--navy-deep);
    padding: 26px 30px;
    color: white;
    position: relative;
}
.pass-stub {
    flex: 1;
    background: var(--navy-panel);
    padding: 26px 20px;
    color: #C9D8EA;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12.5px;
    line-height: 2.1;
    border-right: 2px dashed rgba(255,255,255,0.18);
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.pass-stub b { color: white; display: block; font-size: 13.5px; }
.status-line {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 26px;
    font-weight: 700;
    margin-bottom: 4px;
}
.status-satisfied { color: var(--green); }
.status-dissatisfied { color: var(--red); }
.conf-number {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 44px;
    font-weight: 600;
    color: var(--amber);
    letter-spacing: 1px;
}
.conf-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 1.5px;
    color: #8FA3BF;
}

/* ---- Sidebar stat rows ---- */
.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 13.5px;
    padding: 5px 0;
    border-bottom: 1px solid #EEF2F7;
}
.stat-row span { color: var(--slate); }
.stat-row b {
    font-family: 'IBM Plex Mono', monospace;
    color: var(--navy-deep);
}

@media (prefers-reduced-motion: reduce) {
    .hero, .pass-wrap { animation: none !important; }
}
</style>
""", unsafe_allow_html=True)

# =========================================================================
# HERO
# =========================================================================
st.markdown(f"""
<div class="hero">
    <div class="board-strip">
        <span><span class="dot"></span>MODEL · XGBOOST (TUNED)</span>
        <span>ACCURACY · 95.75%</span>
        <span>DATASET · 129,880 PASSENGERS</span>
    </div>
    <h1>✈️ Flight Satisfaction Predictor</h1>
    <p>Enter a passenger's trip and service ratings below. The model reads them the way a gate agent
    reads a boarding pass — instantly — and returns a satisfaction call with a confidence score.</p>
</div>
""", unsafe_allow_html=True)

# =========================================================================
# SIDEBAR
# =========================================================================
with st.sidebar:
    st.markdown("### ✈️ About this model")
    st.markdown(
        "A tuned **XGBoost** classifier trained on 129,880 airline passenger "
        "records, wrapped in a full preprocessing pipeline (imputation, scaling, "
        "encoding) to avoid data leakage."
    )
    st.markdown("**Test-set performance**")
    st.markdown("""
    <div class="stat-row"><span>Accuracy</span><b>95.75%</b></div>
    <div class="stat-row"><span>F1-score</span><b>96.08%</b></div>
    <div class="stat-row"><span>ROC-AUC</span><b>99.41%</b></div>
    <div class="stat-row"><span>Precision</span><b>97.05%</b></div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown(
        "**Top satisfaction drivers** (via SHAP):\n"
        "1. Seat comfort\n"
        "2. Inflight entertainment\n"
        "3. Customer loyalty status\n"
        "4. Type of travel"
    )
    st.divider()
    st.caption("NTI Summer Training — Final Project")

# =========================================================================
# FORM
# =========================================================================
st.markdown('<div class="section-label">PASSENGER &nbsp;/&nbsp; FLIGHT DETAILS</div>', unsafe_allow_html=True)

with st.form("prediction_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
        age = st.number_input("Age", min_value=1, max_value=100, value=35)
    with c2:
        type_of_travel = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
        travel_class = st.selectbox("Class", ["Business", "Eco", "Eco Plus"])
    with c3:
        flight_distance = st.number_input("Flight Distance (miles)", min_value=0, value=500)

    d1, d2 = st.columns(2)
    with d1:
        departure_delay = st.number_input("Departure Delay in Minutes", min_value=0, value=0)
    with d2:
        arrival_delay = st.number_input("Arrival Delay in Minutes", min_value=0, value=0)

    st.markdown('<div class="section-label">SERVICE RATINGS &nbsp;/&nbsp; 0 = WORST · 5 = BEST</div>', unsafe_allow_html=True)
    ratings = {}
    rating_cols_layout = st.columns(2)
    for i, col_name in enumerate(RATING_COLS):
        with rating_cols_layout[i % 2]:
            ratings[col_name] = st.slider(col_name, 0, 5, 3)

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Predict Satisfaction  →", use_container_width=True)

# =========================================================================
# PREDICTION + BOARDING-PASS RESULT
# =========================================================================
if submitted:
    raw_input = {
        "Customer Type": customer_type,
        "Age": age,
        "Type of Travel": type_of_travel,
        "Class": travel_class,
        "Flight Distance": flight_distance,
        "Departure Delay in Minutes": departure_delay,
        "Arrival Delay in Minutes": arrival_delay,
        **ratings,
    }

    X_input = build_feature_row(raw_input)

    pred_encoded = model.predict(X_input)[0]
    pred_proba = model.predict_proba(X_input)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]
    confidence = float(np.max(pred_proba)) * 100

    is_satisfied = pred_label == "satisfied"
    status_class = "status-satisfied" if is_satisfied else "status-dissatisfied"
    status_text = "SATISFIED" if is_satisfied else "DISSATISFIED"
    status_icon = "😊" if is_satisfied else "😞"

    st.markdown('<div class="section-label">RESULT</div>', unsafe_allow_html=True)

    counter_placeholder = st.empty()

    pass_html = f"""
    <div class="pass-wrap">
        <div class="pass-main">
            <div class="status-line {status_class}">{status_icon}&nbsp; {status_text}</div>
            <div class="conf-label">CONFIDENCE</div>
            <div class="conf-number">{{VALUE}}%</div>
        </div>
        <div class="pass-stub">
            <b>{travel_class.upper()}</b>CLASS
            <b>{type_of_travel.upper()}</b>TRAVEL TYPE
            <b>{customer_type.upper()}</b>LOYALTY
        </div>
    </div>
    """

    # Split-flap style counting animation up to the confidence score
    steps = 18
    for i in range(steps + 1):
        val = round(confidence * i / steps)
        counter_placeholder.markdown(pass_html.replace("{VALUE}", str(val)), unsafe_allow_html=True)
        time.sleep(0.02)
    counter_placeholder.markdown(pass_html.replace("{VALUE}", f"{confidence:.1f}"), unsafe_allow_html=True)

    if is_satisfied:
        components.html("""
        <div style="width:100%;height:190px;overflow:hidden;position:relative;">
        <canvas id="confetti-canvas" style="position:absolute;top:0;left:0;width:100%;height:100%;"></canvas>
        </div>
        <script>
        const canvas = document.getElementById('confetti-canvas');
        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        const w = canvas.clientWidth, h = 190;
        canvas.width = w * dpr; canvas.height = h * dpr;
        ctx.scale(dpr, dpr);
        const colors = ['#FFB300', '#2FBF71', '#FFFFFF', '#7FB8E0'];
        let particles = Array.from({length: 110}, () => ({
            x: Math.random()*w, y: -20 - Math.random()*80,
            r: 3+Math.random()*4, c: colors[Math.floor(Math.random()*colors.length)],
            vy: 2.5+Math.random()*3, vx: -1.5+Math.random()*3, rot: Math.random()*360, vr: -6+Math.random()*12
        }));
        let frame = 0;
        function draw() {
            ctx.clearRect(0,0,w,h);
            particles.forEach(p => {
                p.y += p.vy; p.x += p.vx; p.rot += p.vr;
                ctx.save(); ctx.translate(p.x,p.y); ctx.rotate(p.rot*Math.PI/180);
                ctx.fillStyle = p.c; ctx.fillRect(-p.r/2,-p.r/2,p.r,p.r*0.55); ctx.restore();
            });
            frame++;
            if (frame < 130) requestAnimationFrame(draw);
        }
        draw();
        </script>
        """, height=190)

    with st.expander("See engineered features used by the model"):
        st.dataframe(X_input.T.rename(columns={0: "Value"}), use_container_width=True)

st.markdown(
    "<div style='text-align:center; color:#8FA3BF; font-size:12.5px; margin-top:40px;'>"
    "Model: tuned XGBoost pipeline · Built with Streamlit</div>",
    unsafe_allow_html=True,
)
