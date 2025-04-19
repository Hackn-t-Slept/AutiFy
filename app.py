import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Autify - Autism Screening", page_icon="🧠", layout="centered")

# Load model
@st.cache_resource
def load_model():
    try:
        return joblib.load("autism_prediction_model.pkl")
    except:
        st.error("Model file not found. Please ensure 'autism_prediction_model.pkl' is in the same folder.")
        return None

model = load_model()

# Show logo
st.markdown(
    """
    <div style="text-align: center;">
        <img src="Logo.png" alt="Logo" width="150">
    </div>
    """,
    unsafe_allow_html=True
)

# Dark mode toggle
dark_mode = st.toggle("🌙 Toggle Dark Mode")

# Custom CSS Styling
def inject_custom_css(dark=False):
    st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
        transition: all 0.4s ease;
    }
    .stApp {
        background-color: %s;
        color: %s;
    }
    .card {
        background-color: %s;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    .result {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
    }
    .stButton button {
        border-radius: 8px;
        padding: 10px 20px;
    }
    </style>
    """ % (
        "#1E1E1E" if dark else "#F7F7F7",  # background
        "#FAFAFA" if dark else "#111",    # text
        "#2D2D2D" if dark else "#FFFFFF"  # card bg
    ), unsafe_allow_html=True)

inject_custom_css(dark=dark_mode)

# Title & Description
st.markdown("<h1 style='text-align: center;'>🧠 Autify</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Quick AQ-10 Based Autism Screening</p>", unsafe_allow_html=True)

st.divider()

# Input Form
with st.form("autism_form"):
    st.markdown("#### 🧩 AQ-10 Questions")

    col1, col2 = st.columns(2)
    with col1:
        A1 = st.selectbox("1. I often notice small sounds when others do not.", [0, 1])
        A2 = st.selectbox("2. I usually concentrate more on the whole picture than the details.", [0, 1])
        A3 = st.selectbox("3. I find it easy to do more than one thing at once.", [0, 1])
        A4 = st.selectbox("4. If there is an interruption, I can switch back easily.", [0, 1])
        A5 = st.selectbox("5. I find it easy to ‘read between the lines’.", [0, 1])
    
    with col2:
        A6 = st.selectbox("6. I know if someone is getting bored while I’m talking.", [0, 1])
        A7 = st.selectbox("7. I can imagine characters while reading stories.", [0, 1])
        A8 = st.selectbox("8. I understand feelings by facial expression.", [0, 1])
        A9 = st.selectbox("9. I find it hard to understand others' intentions.", [0, 1])
        A10 = st.selectbox("10. I enjoy social situations.", [0, 1])

    st.divider()
    st.markdown("#### 👤 Personal Information")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 1, 100, 25)
        gender = st.radio("Gender", ['m', 'f'], horizontal=True)
        jaundice = st.radio("Born with jaundice?", ['yes', 'no'], horizontal=True)
        family_history = st.radio("Family history of ASD?", ['yes', 'no'], horizontal=True)

    with col2:
        result = st.slider("AQ Test Result Score (0–10 recommended)", 0, 20, 6)
        used_app_before = st.radio("Used this app before?", ['yes', 'no'], horizontal=True)
        austim = st.radio("Diagnosed with autism before?", ['yes', 'no'], horizontal=True)

    st.divider()
    st.markdown("#### 🌍 Background")

    col1, col2 = st.columns(2)
    with col1:
        ethnicity = st.selectbox("Ethnicity", ['White', 'Latino', 'Black', 'Asian', 'Middle Eastern', 'Pasifika', 'South Asian', 'Hispanic', 'Turkish', 'Others'])
    with col2:
        country = st.selectbox("Country of Residence", ['USA', 'UK', 'India', 'Canada', 'Others'])

    relation = st.selectbox("Who is completing this test?", ['Self', 'Parent', 'Relative', 'Health care professional', 'Others'])

    st.markdown(" ")
    submit = st.form_submit_button("🔍 Predict")

# Predict
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

    try:
        input_data = input_data[model.feature_names_in_]
        prediction = model.predict(input_data)[0]
        result_text = "🔴 Autism Positive" if prediction == 1 else "🟢 Not Autism Positive"
        
        st.markdown(f"""
        <div class="card result">
            {result_text}
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Something went wrong during prediction: {e}")
