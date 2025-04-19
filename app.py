import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load('autism_prediction_model.pkl')

# Page setup
st.set_page_config(page_title="ASD Predictor", page_icon="🧠", layout="centered")

# Custom CSS for style
st.markdown("""
    <style>
        .big-font {
            font-size: 24px !important;
        }
        .result-box {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            font-size: 22px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("## 🧠 Autism Spectrum Disorder (ASD) Prediction")
st.write("Welcome! This AI-powered tool helps you identify traits associated with Autism Spectrum Disorder using the **AQ-10 Screening**.")

st.markdown("### 👇 Please answer the questions below:")

with st.form("autism_form"):
    st.subheader("📋 AQ-10 Questionnaire")

    col1, col2 = st.columns(2)

    with col1:
        A1 = st.radio("1. Notice small sounds others don’t?", [1, 0], index=1, horizontal=True)
        A2 = st.radio("2. Focus on whole picture more than details?", [1, 0], index=1, horizontal=True)
        A3 = st.radio("3. Easy to multitask?", [1, 0], index=1, horizontal=True)
        A4 = st.radio("4. Can switch back easily after interruptions?", [1, 0], index=1, horizontal=True)
        A5 = st.radio("5. Can ‘read between the lines’ in conversation?", [1, 0], index=1, horizontal=True)

    with col2:
        A6 = st.radio("6. Can tell if someone’s bored listening to you?", [1, 0], index=1, horizontal=True)
        A7 = st.radio("7. Easily imagine story characters visually?", [1, 0], index=1, horizontal=True)
        A8 = st.radio("8. Can tell thoughts or feelings from faces?", [1, 0], index=1, horizontal=True)
        A9 = st.radio("9. Find it hard to work out people’s intentions?", [1, 0], index=1, horizontal=True)
        A10 = st.radio("10. Enjoy social situations?", [1, 0], index=1, horizontal=True)

    st.markdown("---")
    st.subheader("🙋 Personal Information")

    col3, col4 = st.columns(2)
    with col3:
        age = st.slider("Select your age", 1, 100, 25)
        gender = st.radio("Gender", ['Male', 'Female'], horizontal=True)

    with col4:
        jaundice = st.radio("Born with jaundice?", ['Yes', 'No'], horizontal=True)
        family_history = st.radio("Family history of ASD?", ['Yes', 'No'], horizontal=True)

    used_app_before = st.radio("Used a screening app before?", ['Yes', 'No'], horizontal=True)

    submit = st.form_submit_button("🔍 Predict")

# Handle submission
if submit:
    try:
        # Map user-friendly answers to model-friendly data
        input_data = pd.DataFrame([{
            "A1_Score": A1, "A2_Score": A2, "A3_Score": A3, "A4_Score": A4,
            "A5_Score": A5, "A6_Score": A6, "A7_Score": A7, "A8_Score": A8,
            "A9_Score": A9, "A10_Score": A10,
            "age": age,
            "gender": 0 if gender == 'Male' else 1,
            "jaundice": 1 if jaundice == 'Yes' else 0,
            "family_history": 1 if family_history == 'Yes' else 0,
            "used_app_before": 1 if used_app_before == 'Yes' else 0
        }])

        # Ensure correct feature order
        input_data = input_data[model.feature_names_in_]

        # Make prediction
        prediction = model.predict(input_data)[0]

        if prediction == 0:
            st.markdown('<div class="result-box">🟢 <span style="color:green">No ASD traits detected.</span><br>Based on your responses, you do not show signs of ASD traits.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box">🔴 <span style="color:red">Possible ASD traits detected.</span><br>It might be helpful to consult a healthcare professional for further assessment.</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Something went wrong during prediction. ❌\n\n**Error:** {e}")
