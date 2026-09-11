import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Airline Customer Satisfaction Predictor",
    page_icon="✈️",
    layout="wide"
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


# -----------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------
st.title("✈️ Airline Customer Satisfaction Predictor")
st.markdown(
    "Enter a passenger's flight and service details to predict whether "
    "they will be **satisfied** or **dissatisfied**, powered by a tuned "
    "XGBoost model."
)
st.divider()

# -----------------------------------------------------------------------
# Input form
# -----------------------------------------------------------------------
with st.form("prediction_form"):
    st.subheader("Passenger & Flight Details")
    c1, c2, c3 = st.columns(3)
    with c1:
        customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
        age = st.number_input("Age", min_value=1, max_value=100, value=35)
    with c2:
        type_of_travel = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
        travel_class = st.selectbox("Class", ["Business", "Eco", "Eco Plus"])
    with c3:
        flight_distance = st.number_input("Flight Distance (miles)", min_value=0, value=500)

    st.subheader("Delays")
    d1, d2 = st.columns(2)
    with d1:
        departure_delay = st.number_input("Departure Delay in Minutes", min_value=0, value=0)
    with d2:
        arrival_delay = st.number_input("Arrival Delay in Minutes", min_value=0, value=0)

    st.subheader("Service Ratings (0 = worst, 5 = best)")
    ratings = {}
    rating_cols_layout = st.columns(2)
    for i, col_name in enumerate(RATING_COLS):
        with rating_cols_layout[i % 2]:
            ratings[col_name] = st.slider(col_name, 0, 5, 3)

    submitted = st.form_submit_button("Predict Satisfaction", use_container_width=True)

# -----------------------------------------------------------------------
# Prediction
# -----------------------------------------------------------------------
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

    st.divider()
    st.subheader("Prediction Result")

    r1, r2 = st.columns([1, 1.5])
    with r1:
        if pred_label == "satisfied":
            st.success(f"### 😊 {pred_label.title()}")
        else:
            st.error(f"### 😞 {pred_label.title()}")

    with r2:
        proba_df = pd.DataFrame(
            {"Class": label_encoder.classes_, "Probability": pred_proba}
        ).sort_values("Probability", ascending=False)
        st.dataframe(
            proba_df.style.format({"Probability": "{:.1%}"}),
            hide_index=True,
            use_container_width=True,
        )

    with st.expander("See engineered features used by the model"):
        st.dataframe(X_input.T.rename(columns={0: "Value"}), use_container_width=True)

st.divider()
st.caption("Model: tuned XGBoost pipeline · Built with Streamlit")
