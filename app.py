import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load and prepare model (for simplicity, train here — or load .pkl if saved)
@st.cache_data
def load_model():
    df = pd.read_csv("train.csv")
    X = df.drop(columns=['ID', 'Class/ASD'])
    y = df['Class/ASD']
    
    # Encode categorical columns
    X = X.apply(lambda col: LabelEncoder().fit_transform(col) if col.dtype == 'object' else col)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model, X.columns

model, feature_names = load_model()

# Streamlit UI
st.title("🧠 Autism Prediction App")
st.write("Fill out the form to check the autism prediction.")

# Input form
user_input = {}
for feature in feature_names:
    if feature in ['age', 'result']:
        user_input[feature] = st.number_input(f"{feature.capitalize()}", min_value=0.0, max_value=100.0)
    else:
        user_input[feature] = st.text_input(f"{feature.capitalize()}")

if st.button("Predict"):
    try:
        input_df = pd.DataFrame([user_input])
        # Encode categorical fields same as during training
        input_df = input_df.apply(lambda col: LabelEncoder().fit_transform(col) if col.dtype == 'object' else col)
        prediction = model.predict(input_df)[0]
        if prediction == 1:
            st.error("🔴 Autism Positive")
        else:
            st.success("🟢 Not Autism Positive")
    except Exception as e:
        st.warning("Error in input. Please check the values.")
