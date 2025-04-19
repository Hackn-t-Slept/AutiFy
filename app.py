import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load('autism_model.pkl')

st.title("Autism Spectrum Disorder (ASD) Prediction")

# Input form
with st.form("autism_form"):
    A1_Score = st.selectbox("A1 Score", [0, 1])
    A2_Score = st.selectbox("A2 Score", [0, 1])
    A3_Score = st.selectbox("A3 Score", [0, 1])
    A4_Score = st.selectbox("A4 Score", [0, 1])
    A5_Score = st.selectbox("A5 Score", [0, 1])
    A6_Score = st.selectbox("A6 Score", [0, 1])
    A7_Score = st.selectbox("A7 Score", [0, 1])
    A8_Score = st.selectbox("A8 Score", [0, 1])
    A9_Score = st.selectbox("A9 Score", [0, 1])
    A10_Score = st.selectbox("A10 Score", [0, 1])

    age = st.slider("Age", 1, 100, 25)
    gender = st.selectbox("Gender", ['m', 'f'])
    jaundice = st.selectbox("Born with jaundice?", ['yes', 'no'])
    family_history = st.selectbox("Family history of ASD?", ['yes', 'no'])
    used_app_before = st.selectbox("Used screening app before?", ['yes', 'no'])

    submit = st.form_submit_button("Predict")

if submit:
    input_data = pd.DataFrame([{
        "A1_Score": A1_Score, "A2_Score": A2_Score, "A3_Score": A3_Score, "A4_Score": A4_Score,
        "A5_Score": A5_Score, "A6_Score": A6_Score, "A7_Score": A7_Score, "A8_Score": A8_Score,
        "A9_Score": A9_Score, "A10_Score": A10_Score, "age": age, "gender": gender,
        "jaundice": jaundice, "family_history": family_history, "used_app_before": used_app_before
    }])

    # Preprocess if needed (encoding, scaling, etc.)
    # Example: label encode gender etc., if your model expects it

    prediction = model.predict(input_data)[0]
    st.success(f"Prediction: {'ASD Positive' if prediction == 1 else 'ASD Negative'}")
