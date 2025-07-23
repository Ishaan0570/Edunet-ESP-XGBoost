import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("xgb_model.pkl")

st.set_page_config(page_title="Income Class Prediction", page_icon="💰", layout="centered")

st.title("💰 Income Class Prediction App")
st.markdown("Predict whether a person earns >50K or ≤50K based on census features.")

# Sidebar input
st.sidebar.header("Input Features")

# Feature 1: Age (numerical)
age = st.sidebar.slider("Age", 18, 90, 35)

# Feature 2: Education Level -> mapped to 'educational-num'
edu_map = {
    "Preschool": 1,
    "1st-4th": 2,
    "5th-6th": 3,
    "7th-8th": 4,
    "9th": 5,
    "10th": 6,
    "11th": 7,
    "12th": 8,
    "HS-grad": 9,
    "Some-college": 10,
    "Assoc-voc": 11,
    "Assoc-acdm": 12,
    "Bachelors": 13,
    "Masters": 14,
    "Prof-school": 15,
    "Doctorate": 16
}
education_label = st.sidebar.selectbox("Education Level", list(edu_map.keys()))
edu_num = edu_map[education_label]

# Feature 3: Marital Status (categorical -> encoded)
marital_status_map = {
    "Married-civ-spouse": 0,
    "Divorced": 1,
    "Never-married": 2,
    "Separated": 3,
    "Others": 4
}
marital_status_label = st.sidebar.selectbox("Marital Status", list(marital_status_map.keys()))
marital_status = marital_status_map[marital_status_label]

# Feature 4: Relationship (categorical -> encoded)
relationship_map = {
    "Husband": 0,
    "Not-in-family": 1,
    "Own-child": 2,
    "Unmarried": 3,
    "Wife": 4,
    "Others": 5
}
relationship_label = st.sidebar.selectbox("Relationship", list(relationship_map.keys()))
relationship = relationship_map[relationship_label]

# Feature 5: Hours per week (numerical)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 99, 40)

# Build input DataFrame
input_df = pd.DataFrame({
    'age': [age],
    'educational-num': [edu_num],
    'marital-status': [marital_status],
    'relationship': [relationship],
    'hours-per-week': [hours_per_week]
})

st.write("### 🔎 Input Data")
st.write(input_df)

# Predict button
if st.button("Predict Income Class"):
    prediction = model.predict(input_df)
    label = ">50K" if prediction[0] == 1 else "≤50K"
    st.success(f"✅ Prediction: {label}")

# Batch prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())

    required_columns = ['age', 'educational-num', 'marital-status', 'relationship', 'hours-per-week']
    if all(col in batch_data.columns for col in required_columns):
        batch_preds = model.predict(batch_data[required_columns])
        batch_data['PredictedClass'] = np.where(batch_preds == 1, '>50K', '≤50K')
        st.write("✅ Predictions:")
        st.write(batch_data.head())

        csv = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions CSV", csv, file_name='predicted_income.csv', mime='text/csv')
    else:
        st.error("❌ CSV is missing one or more required columns.")
