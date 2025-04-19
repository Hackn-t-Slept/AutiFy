import streamlit as st
import pandas as pd
import joblib

# Load the trained model
@st.cache_resource
def load_model():
    try:
        return joblib.load("autism_prediction_model.pkl")
    except:
        st.error("🚫 Model file not found. Make sure 'autism_prediction_model.pkl' is available.")
        return None

model = load_model()

# Page setup
st.set_page_config(page_title="🧠 Autify - ASD Predictor", layout="centered")
st.markdown(
    """
    <style>
    .main { background-color: #f4f4f8; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align: center; color: #4B0082;'>🧠 Autify</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Your personal ASD Screening Assistant</h4>", unsafe_allow_html=True)

st.markdown("---")

# Input form
with st.form("autism_form"):
    st.subheader("📋 AQ-10 Questions")

    col1, col2 = st.columns(2)
    with col1:
        A1 = st.radio("1️⃣ Notice small sounds others don't?", [0, 1], horizontal=True)
        A2 = st.radio("2️⃣ Focus more on big picture than details?", [0, 1], horizontal=True)
        A3 = st.radio("3️⃣ Easy to multitask?", [0, 1], horizontal=True)
        A4 = st.radio("4️⃣ Can resume after interruption?", [0, 1], horizontal=True)
        A5 = st.radio("5️⃣ Understand implied meanings?", [0, 1], horizontal=True)
    with col2:
        A6 = st.radio("6️⃣ Know if someone is bored?", [0, 1], horizontal=True)
        A7 = st.radio("7️⃣ Imagine story characters easily?", [0, 1], horizontal=True)
        A8 = st.radio("8️⃣ Recognize feelings from faces?", [0, 1], horizontal=True)
        A9 = st.radio("9️⃣ Find it hard to understand intentions?", [0, 1], horizontal=True)
        A10 = st.radio("🔟 Enjoy social situations?", [0, 1], horizontal=True)

    st.markdown("---")
    st.subheader("🧍 Demographics")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("📅 Age", 1, 100, 25)
        result = st.slider("🧪 AQ Test Score (0–20)", 0, 20, 6)
        gender = st.selectbox("👤 Gender", ['m', 'f'])
        jaundice = st.selectbox("🍼 Born with jaundice?", ['yes', 'no'])
        family_history = st.selectbox("🧬 Family history of ASD?", ['yes', 'no'])
    with col2:
        used_app_before = st.selectbox("📱 Used screening app before?", ['yes', 'no'])
        austim = st.selectbox("📖 Prior ASD diagnosis?", ['yes', 'no'])
        ethnicity = st.selectbox("🌍 Ethnicity", [
            'White', 'Latino', 'Black', 'Asian', 'Middle Eastern', 'Pasifika',
            'South Asian', 'Hispanic', 'Turkish', 'Others'
        ])
        country = st.selectbox("🏡 Country of Residence", ['USA', 'UK', 'India', 'Canada', 'Others'])
        relation = st.selectbox("🤝 Relation of respondent", [
            'Self', 'Parent', 'Relative', 'Health care professional', 'Others'
        ])

    submit = st.form_submit_button("🔍 Predict")

if submit and model:
    input_data = pd.DataFrame([{
        'A1_Score': A1, 'A2_Score': A2, 'A3_Score': A3, 'A4_Score': A4, 'A5_Score': A5,
        'A6_Score': A6, 'A7_Score': A7, 'A8_Score': A8, 'A9_Score': A9, 'A10_Score': A10,
        'age': age, 'result': result,
        'gender': gender, 'jaundice': jaundice, 'used_app_before': used_app_before,
        'ethnicity': ethnicity, 'austim': austim,
        'contry_of_res': country, 'relation': relation,
        'family_history': family_history
    }])

    # Encoding categorical features
    try:
        input_data['gender'] = input_data['gender'].map({'m': 0, 'f': 1})
        input_data['jaundice'] = input_data['jaundice'].map({'no': 0, 'yes': 1})
        input_data['family_history'] = input_data['family_history'].map({'no': 0, 'yes': 1})
        input_data['used_app_before'] = input_data['used_app_before'].map({'no': 0, 'yes': 1})
        input_data['austim'] = input_data['austim'].map({'no': 0, 'yes': 1})
        input_data['relation'] = input_data['relation'].map({
            'Self': 0, 'Parent': 1, 'Relative': 2, 'Health care professional': 3, 'Others': 4
        })
        input_data['ethnicity'] = input_data['ethnicity'].astype('category').cat.codes
        input_data['contry_of_res'] = input_data['contry_of_res'].astype('category').cat.codes

        # Match model input columns
        input_data = input_data[model.feature_names_in_]

        prediction = model.predict(input_data)[0]
        if prediction == 1:
            st.markdown("### 🧠 Result: <span style='color:red'>
