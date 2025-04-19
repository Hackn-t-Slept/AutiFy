import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import base64
from io import BytesIO

st.set_page_config(page_title="Autify - Autism Screening", page_icon="🧠", layout="centered")

# Function to convert image to base64
def image_to_bytes(image: Image.Image) -> bytes:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue())

st.image_to_bytes = image_to_bytes  # Simple monkey patch

# Load model
@st.cache_resource
def load_model():
    try:
        return joblib.load("autism_prediction_model.pkl")
    except:
        st.error("Model file not found. Please ensure 'autism_prediction_model.pkl' is in the same folder.")
        return None

model = load_model()

# Custom CSS Styling
def inject_custom_css():
    bg_gradient = """
        linear-gradient(135deg, #310E68, #4F2E7D);
    """

    st.markdown(f"""
    <style>
    html, body, [class*="css"] {{
        font-family: 'Segoe UI', sans-serif;
        transition: all 0.4s ease;
    }}
    .stApp {{
        background: {bg_gradient};
        color: #FAFAFA;
        animation: fadeIn 1.2s ease-in;
        border: 5px solid #6A3E98; /* Adds a border around the entire page */
        border-radius: 16px; /* Makes the edges rounded */
        padding: 15px; /* Adds spacing inside the border */
        margin: 10px; /* Adds spacing between the border and viewport */
    }}
    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to {{ opacity: 1; }}
    }}
    .card {{
        background-color: #fff;
        border: 2px solid #6A3E98; /* Border for individual card components */
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        margin-top: 20px;
        transition: all 0.3s ease;
    }}
    .card:hover {{
        transform: scale(1.01);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }}
    .result {{
        font-size: 1.6rem;
        font-weight: bold;
        text-align: center;
        padding-top: 10px;
        padding-bottom: 10px;
    }}
    .stButton button {{
        border: 2px solid #6849A2; /* Border for buttons */
        border-radius: 8px;
        padding: 10px 20px;
        background-color: #7C5CB9;
        color: white;
        font-weight: bold;
    }}
    .stButton button:hover {{
        background-color: #6849A2;
        border-color: #563885; /* Darker border on hover */
    }}
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# Logo and Title Side-by-Side
logo = Image.open("Logo.png")
logo_b64 = st.image_to_bytes(logo).decode('utf-8')
st.markdown(f"""
<div style='display: flex; align-items: center; justify-content: center; gap: 20px; margin-bottom: 20px;'>
    <img src='data:image/png;base64,{logo_b64}' style='height: 80px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.2);'/>
    <div>
        <h1 style='margin: 0;'>Autify</h1>
        <p style='margin: 0;'>We Connect the Dots—Even the Ones You Didn’t See</p>
    </div>
</div>
""", unsafe_allow_html=True)

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

# Prediction Logic
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
        """,
