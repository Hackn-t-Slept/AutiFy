import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("autism_prediction_model.pkl")
        return model
    except:
        st.error("Model file not found. Please ensure 'autism_prediction_model.pkl' is in the same folder.")
        return None

model = load_model()

st.title("🧠 Autify")
st.write("Answer the questions below to check for ASD traits (based on AQ-10 screening).")

# Input form
with st.form("autism_form"):
    A1 = st.selectbox("1. I often notice small sounds when others do not.", [0, 1])
    A2 = st.selectbox("2. I usually concentrate more on the whole picture than the details.", [0, 1])
    A3 = st.selectbox("3. I find it easy to do more than one thing at once.", [0, 1])
    A4 = st.selectbox("4. If there is an interruption, I can switch back easily.", [0, 1])
    A5 = st.selectbox("5. I find it easy to ‘read between the lines’ when someone is talking to me.", [0, 1])
    A6 = st.selectbox("6. I know how to tell if someone listening to me is getting bored.", [0, 1])
    A7 = st.selectbox("7. When I’m reading a story, I can easily imagine what the characters might look like.", [0, 1])
    A8 = st.selectbox("8. I find it easy to work out what someone is thinking or feeling just by looking at their face.", [0, 1])
    A9 = st.selectbox("9. I find it difficult to work out people’s intentions.", [0, 1])
    A10 = st.selectbox("10. I enjoy social situations.", [0, 1])

    st.markdown("---")
    age = st.slider("Age", 1, 100, 25)
    result = st.slider("AQ Test Result Score (0-10 recommended)", 0, 20, 6)
    gender = st.selectbox("Gender", ['m', 'f'])
    jaundice = st.selectbox("Were you born with jaundice?", ['yes', 'no'])
    family_history = st.selectbox("Is there a family history of ASD?", ['yes', 'no'])
    used_app_before = st.selectbox("Used a screening app before?", ['yes', 'no'])

    st.markdown("---")
    ethnicity = st.selectbox("Ethnicity", ['White', 'Latino', 'Black', 'Asian', 'Middle Eastern', 'Pasifika', 'South Asian', 'Hispanic', 'Turkish', 'Others'])
    austim = st.selectbox("Diagnosed with autism before?", ['yes', 'no'])
    country = st.selectbox("Country of Residence", ['USA', 'UK', 'India', 'Canada', 'Others'])
    relation = st.selectbox("Relation of person completing the test", ['Self', 'Parent', 'Relative', 'Health care professional', 'Others'])

    submit = st.form_submit_button("Predict")

if submit and model:
    input_data = pd.DataFrame([{
        'A1_Score': A1, 'A2_Score': A2, 'A3_Score': A3, 'A4_Score': A4, 'A5_Score': A5,
        'A6_Score': A6, 'A7_Score': A7, 'A8_Score': A8, 'A9_Score': A9, 'A10_Score': A10,
        'age': age, 'result': result,
        'gender': gender, 'jaundice': jaundice, 'used_app_before': used_app_before,
        'ethnicity': ethnicity, 'austim': austim,
        'contry_of_res': country, 'relation': relation,
        'family_history': family_history  # if model uses this
    }])

    # Encode categorical values to match training
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

    # Align features
    try:
        input_data = input_data[model.feature_names_in_]
        prediction = model.predict(input_data)[0]
        result_text = "🔴 Autism Positive" if prediction == 1 else "🟢 Not Autism Positive"
        st.success(f"Prediction: {result_text}")
    except Exception as e:
        st.error(f"Something went wrong: {e}")
