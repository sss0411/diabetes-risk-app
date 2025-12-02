import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("diabetes_model.joblib")
    scaler = joblib.load("scaler.joblib")
    return model, scaler

model, scaler = load_artifacts()


st.title("🩺 Diabetes Risk Prediction App")
st.write("Machine learning model for predicting diabetes risk based on clinical & lifestyle factors.")

# -----------------------------
# Input fields
# -----------------------------
st.header("Patient Information")

age = st.number_input("Age", 10, 100, 40)
bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
f_glucose = st.number_input("Fasting Glucose (mg/dL)", 50, 300, 100)
blood_pressure = st.number_input("Blood Pressure", 80, 200, 120)
hba1c = st.number_input("HbA1c (%)", 3.0, 15.0, 5.5)
daily_cal = st.number_input("Daily Calories", 500, 5000, 2000)
phys_act = st.number_input("Physical Activity (min/day)", 0, 300, 30)
smoke = st.selectbox("Smoking Status", ["No", "Yes"])
family = st.selectbox("Family History of Diabetes", ["No", "Yes"])

# Encoding
smoke = 1 if smoke == "Yes" else 0
family = 1 if family == "Yes" else 0

# -----------------------------
# Prepare input
# -----------------------------
input_data = pd.DataFrame({
    "Age": [age],
    "BMI": [bmi],
    "Fasting_Glucose": [f_glucose],
    "Blood_Pressure": [blood_pressure],
    "HbA1c": [hba1c],
    "Daily_Calories": [daily_cal],
    "Physical_Activity_min_per_day": [phys_act],
    "Smoking_Status": [smoke],
    "Family_History": [family]
})

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict Diabetes Risk"):
    scaled_input = scaler.transform(input_data)
risk = model.predict_proba(scaled_input)[0][1]

    st.subheader(f"Predicted Diabetes Risk: {risk:.3f}")

    # -----------------------------
    # Risk category
    # -----------------------------
    if risk < 0.20:
        category = "🟢 Low risk"
    elif risk < 0.40:
        category = "🟡 Medium risk"
    elif risk < 0.70:
        category = "🟠 High risk"
    else:
        category = "🔴 Very high risk"

    st.write("### Risk category:", category)

    # -----------------------------
    # Feature importance (light text version)
    # -----------------------------
    st.write("### Key contributing factors")

    importances = model.feature_importances_
    ranked = sorted(
        zip(importances, input_data.columns),
        key=lambda x: x[0],
        reverse=True
    )

    for imp, name in ranked[:5]:
        st.write(f"**{name}** — importance: {imp:.3f}")

    st.info("This is a simplified explanation based on Random Forest feature importances.")

