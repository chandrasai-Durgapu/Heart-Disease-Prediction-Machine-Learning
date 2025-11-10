import streamlit as st
import pandas as pd
import requests

# FastAPI backend URL (adjust if running on different host/port)
FASTAPI_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="❤️ Heart Disease Prediction", layout="centered")
st.title("❤️ Heart Disease Prediction App")
st.markdown("Enter patient details below and get instant predictions!")

# --- Input fields ---
col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", min_value=1, max_value=120, value=40)
    Sex = st.selectbox("Sex", ["M", "F"])
    ChestPainType = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
    RestingBP = st.number_input("Resting BP", min_value=0, max_value=200, value=120)
    Cholesterol = st.number_input("Cholesterol", min_value=0, max_value=600, value=200)

with col2:
    FastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    RestingECG = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    MaxHR = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
    ExerciseAngina = st.selectbox("Exercise Angina", ["Y", "N"])
    Oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    ST_Slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# --- Predict button ---
if st.button("🔍 Predict Heart Disease"):
    # Prepare input data as dict
    input_data = {
        "Age": Age,
        "Sex": Sex,
        "ChestPainType": ChestPainType,
        "RestingBP": RestingBP,
        "Cholesterol": Cholesterol,
        "FastingBS": FastingBS,
        "RestingECG": RestingECG,
        "MaxHR": MaxHR,
        "ExerciseAngina": ExerciseAngina,
        "Oldpeak": Oldpeak,
        "ST_Slope": ST_Slope,
    }

    try:
        response = requests.post(FASTAPI_URL, json=input_data)

        if response.status_code == 200:
            result = response.json()
            if "prediction" in result:
                if result["prediction"] == 1:
                    st.error("🚨 The model predicts **Heart Disease (Positive)**.")
                else:
                    st.success("✅ The model predicts **No Heart Disease (Negative)**.")
            else:
                st.warning(f"⚠️ {result.get('error', 'Unexpected error occurred')}")
        else:
            st.error(f"❌ Server error: {response.status_code}")

    except Exception as e:
        st.error(f"⚠️ Connection Error: {e}")
