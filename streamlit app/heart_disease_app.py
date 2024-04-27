# Import necessary libraries

import streamlit as st
import pandas as pd
import joblib

# load ML Model

model = joblib.load("Heart Disease Predictor Model.joblib")

# create input widgets

Age = st.number_input("Age", min_value=0, max_value=100, value=54) # if no input, then the median age = 54 will be the default

Sex = st.selectbox("Sex", ["M", "F"])

ChestPainType = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])

RestingBP = st.number_input("Resting Blood Pressure", min_value=0, max_value=250, value=130) # if no input, then the median RestingBP = 130 will be the default

Cholesterol = st.number_input("Cholesterol", min_value=0, max_value=650, value=223) # if no input, then the median Cholesterol = 223 will be the default

# Create a dictionary to map user input to integer values
fasting_bs_mapping = {
    "Yes": 1,
    "No": 0
}

# Create the radio button input for fasting blood sugar and map the selected value to an integer
fasting_bs_input = st.radio("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
FastingBS = fasting_bs_mapping[fasting_bs_input]


RestingECG = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])

MaxHR = st.number_input("Max Heart Rate", min_value=50, max_value=250, value=138)  # if no input, then the median MaxHR = 138 will be the default

ExerciseAngina = st.radio("Exercise Induced Angina", ["Y", "N"])

Oldpeak = st.number_input("Oldpeak (ST depression induced by exercise)", min_value=-3.0, max_value=7.0, step=0.1, value=0.6)

ST_Slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# converting input into a pandas dataframe

input_data = pd.DataFrame({
    'Age': [Age],
    'Sex': [Sex],
    'ChestPainType': [ChestPainType],
    'RestingBP': [RestingBP],
    'Cholesterol': [Cholesterol],
    'FastingBS': [FastingBS],
    'RestingECG': [RestingECG],
    'MaxHR': [MaxHR],
    'ExerciseAngina': [ExerciseAngina],
    'Oldpeak': [Oldpeak],
    'ST_Slope': [ST_Slope]
})

if st.button("Predict"):
    prediction = model.predict(input_data)
    st.write(f"Heart Disease Risk Prediction: {'High' if prediction == 1 else 'Low'}")
