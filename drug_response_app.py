import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("models/drug_response_model.pkl")

st.title("💊 Drug Response Prediction App")
st.write("Enter patient details to predict if they will respond to the drug.")

# Collect input features
age = st.number_input("Age", min_value=0, max_value=120, value=50)
sex = st.selectbox("Sex", options=["Male", "Female"])
weight = st.number_input("Weight (kg)", min_value=0.0, value=70.0)
bp = st.slider("Blood Pressure", -5.0, 5.0, 0.0)
chol = st.slider("Cholesterol", -5.0, 5.0, 0.0)
glucose = st.slider("Glucose", -5.0, 5.0, 0.0)
gm1 = st.slider("Genetic Marker 1", -5.0, 5.0, 0.0)
gm2 = st.slider("Genetic Marker 2", -5.0, 5.0, 0.0)
dosage = st.slider("Drug Dosage", -5.0, 5.0, 0.0)
duration = st.slider("Drug Duration", -5.0, 5.0, 0.0)
prev_cond = st.slider("Previous Conditions", -5.0, 5.0, 0.0)
liver_score = st.slider("Liver Function Score", -5.0, 5.0, 0.0)

# Map sex to numeric
sex_numeric = 1 if sex == "Male" else 0

# Predict button
if st.button("Predict Drug Response"):
    input_data = np.array([[age, sex_numeric, weight, bp, chol, glucose, gm1, gm2,
                            dosage, duration, prev_cond, liver_score]])
    
    prediction = model.predict(input_data)[0]
    result = "🟢 Likely to Respond" if prediction == 1 else "🔴 Unlikely to Respond"
    st.subheader("Prediction Result:")
    st.success(result)
