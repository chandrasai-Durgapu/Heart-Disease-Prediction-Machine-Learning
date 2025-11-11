import streamlit as st
import requests

# --- Configuration ---
# MUST match the port where your FastAPI server is running
FASTAPI_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("❤️ Heart Disease Prediction")
st.markdown("Enter the patient's data below to get a prediction.")

# --- Input Form ---
with st.form("prediction_form"):
    st.header("Patient Vitals")
    
    # Row 1: Age, Sex, ChestPainType
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 20, 90, 50, key='Age')
    with col2:
        sex = st.selectbox("Sex", ["M", "F"], key='Sex')
    with col3:
        cp_type = st.selectbox("Chest Pain Type", ["ATA", "ASY", "NAP", "TA"], key='ChestPainType')

    # Row 2: RestingBP, Cholesterol, FastingBS
    col4, col5, col6 = st.columns(3)
    with col4:
        resting_bp = st.number_input("Resting BP (mmHg)", 80, 200, 120, key='RestingBP')
    with col5:
        cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200, key='Cholesterol')
    with col6:
        fasting_bs = st.selectbox("Fasting BS > 120 mg/dl?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No", key='FastingBS')

    # Row 3: RestingECG, MaxHR, ExerciseAngina
    col7, col8, col9 = st.columns(3)
    with col7:
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"], key='RestingECG')
    with col8:
        max_hr = st.number_input("Max Heart Rate", 60, 202, 150, key='MaxHR')
    with col9:
        exercise_angina = st.selectbox("Exercise Induced Angina", ["Y", "N"], key='ExerciseAngina')

    # Row 4: Oldpeak, ST_Slope
    col10, col11, _ = st.columns(3)
    with col10:
        oldpeak = st.number_input("Oldpeak", 0.0, 6.2, 1.0, step=0.1, format="%.1f", key='Oldpeak')
    with col11:
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"], key='ST_Slope')
        
    st.markdown("---")
    submitted = st.form_submit_button("Get Prediction")

# --- Prediction Logic (Corrected Error Handling) ---
if submitted:
    # 1. Prepare data payload
    input_data = {
        "Age": age, "Sex": sex, "ChestPainType": cp_type, "RestingBP": resting_bp,
        "Cholesterol": cholesterol, "FastingBS": fasting_bs, "RestingECG": resting_ecg,
        "MaxHR": max_hr, "ExerciseAngina": exercise_angina, "Oldpeak": oldpeak,
        "ST_Slope": st_slope
    }

    # 2. Send POST request to FastAPI
    try:
        response = requests.post(FASTAPI_URL, json=input_data)
        
        if response.status_code == 200:
            result = response.json()
            st.success("✅ Prediction Received!")
            
            # Display result
            # This block is safe because status_code is 200
            if result.get('prediction') == 1:
                st.error(f"🔴 Result: {result.get('result', 'Heart Disease Detected')}", icon="🚨")
                st.balloons()
            else:
                st.success(f"🟢 Result: {result.get('result', 'No Heart Disease')}", icon="👍")
        
        # --- CORRECTED ERROR HANDLING BLOCK ---
        else:
            # Handle non-200 responses (e.g., 422 Validation Error, 500 Server Error)
            st.error(f"❌ API Error (Status {response.status_code})")
            
            try:
                # Attempt to parse the error detail from the JSON response
                error_data = response.json()
                # FastAPI/Pydantic errors use 'detail'
                error_detail = error_data.get('detail', f'Server returned non-JSON error.')
                
                if isinstance(error_detail, list):
                    # For Pydantic validation errors, 'detail' is often a list of errors
                    st.warning("⚠️ Data Validation Failed:")
                    for err in error_detail:
                        st.code(f"Field: {err.get('loc')[-1]}, Message: {err.get('msg')}")
                else:
                    st.warning(f"**Server Message:** {error_detail}")
                    st.info("Check the FastAPI server console for the full Python traceback.")

            except requests.exceptions.JSONDecodeError:
                st.warning("Could not read error message from server response.")
                st.code(response.text)
        # --- END OF CORRECTED BLOCK ---
            
    except requests.exceptions.ConnectionError:
        st.error("🔌 Connection Error: Could not connect to the FastAPI server.")
        st.info(f"Please ensure the server is running at **{FASTAPI_URL.replace('/predict', '')}**.")