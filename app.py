import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load('autism_prediction_model.pkl')

st.title("🧠 Autism Spectrum Disorder (ASD) Prediction App")
st.write("Answer the questions below to check for ASD traits (based on AQ-10 screening).")

# Input form
with st.form("autism_form"):
    A1_Score = st.selectbox("1. I often notice small sounds when others do not.", [0, 1])
    A2_Score = st.selectbox("2. I usually concentrate more on the whole picture than the details.", [0, 1])
    A3_Score = st.selectbox("3. I find it easy to do more than one thing at once.", [0, 1])
    A4_Score = st.selectbox("4. If there is an interruption, I can switch back easily.", [0, 1])
    A5_Score = st.selectbox("5. I find it easy to ‘read between the lines’ when someone is talking to me.", [0, 1])
    A6_Score = st.selectbox("6. I know how to tell if someone listening to me is getting bored.", [0, 1])
    A7_Score = st.selectbox("7. When I’m reading a story, I can easily imagine what the characters might look like.", [0, 1])
    A8_Score = st.selectbox("8. I find it easy to work out what someone is thinking or feeling just by looking at their face.", [0, 1])
    A9_Score = st.selectbox("9. I find it difficult to work out people’s intentions.", [0, 1])
    A10_Score = st.selectbox("10. I enjoy social situations.", [0, 1])

    st.markdown("---")

    age = st.slider("Age", 1, 100, 25)
    gender = st.selectbox("Gender", ['m', 'f'])
    jaundice = st.selectbox("Were you born with jaundice?", ['yes', 'no'])
    family_history = st.selectbox("Is there a family history of ASD?", ['yes', 'no'])
    used_app_before = st.selectbox("Have you used a screening app before?", ['yes', 'no'])

    st.markdown("---")

    ethnicity = st.selectbox("Ethnicity", ['White', 'Latino', 'Black', 'Asian', 'Middle Eastern', 'Pasifika', 'South Asian', 'Hispanic', 'Turkish', 'Others'])
    austim = st.selectbox("Have you been diagnosed with autism before?", ['yes', 'no'])
    contry_of_res = st.selectbox("Country of Residence", ['USA', 'UK', 'India', 'Canada', 'Others'])
    relation = st.selectbox("Relation of person completing the test", ['Self', 'Parent', 'Relative', 'Health care professional', 'Others'])

    submit = st.form_submit_button("Predict")

if submit:
    input_data = pd.DataFrame([{
        "A1_Score": A1_Score, "A2_Score": A2_Score, "A3_Score": A3_Score, "A4_Score": A4_Score,
        "A5_Score": A5_Score, "A6_Score": A6_Score, "A7_Score": A7_Score, "A8_Score": A8_Score,
        "A9_Score": A9_Score, "A10_Score": A10_Score, "age": age, "gender": gender,
        "jaundice": jaundice, "family_history": family_history, "used_app_before": used_app_before,
        "ethnicity": ethnicity, "austim": austim, "contry_of_res": contry_of_res,
        "relation": relation, "result": 0  # Placeholder, assuming 'result' was in original dataset but not a target
    }])

    # Encode categorical variables
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

    # Ensure all expected features are present and ordered correctly
    try:
        input_data = input_data[model.feature_names_in_]
        prediction = model.predict(input_data)[0]
        result = '🟢 ASD Negative' if prediction == 0 else '🔴 ASD Positive'
        st.success(f"Prediction: {result}")
    except Exception as e:
        st.error(f"Prediction failed. Error: {e}")
