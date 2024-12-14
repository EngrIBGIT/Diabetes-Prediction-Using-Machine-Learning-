import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('xgb_model.pkl')

# App title and logo
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="wide")
st.sidebar.image("logo.png", use_container_width=True)  # Ensure logo.png exists in your working directory
st.title("🩺 Diabetes Prediction App")
st.write("This app predicts whether a patient will have diabetes based on input parameters.")

# Sidebar for user input
st.sidebar.header("Input Patient Parameters:")
Pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
PlasmaGlucose = st.sidebar.number_input("Plasma Glucose", min_value=0, max_value=300, value=120)
DiastolicBloodPressure = st.sidebar.number_input("Diastolic Blood Pressure", min_value=0, max_value=200, value=80)
TricepsThickness = st.sidebar.number_input("Triceps Thickness", min_value=0, max_value=100, value=20)
SerumInsulin = st.sidebar.number_input("Serum Insulin", min_value=0, max_value=800, value=80)
BMI = st.sidebar.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
DiabetesPedigree = st.sidebar.number_input("Diabetes Pedigree", min_value=0.0, max_value=2.5, value=0.5, step=0.01)
Age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30)

# Predict button
if st.button("Predict"):
    # Create input array for prediction
    input_data = np.array([[Pregnancies, PlasmaGlucose, DiastolicBloodPressure, TricepsThickness, SerumInsulin, BMI, DiabetesPedigree, Age]])
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    # Output results
    st.write("### Prediction:")
    if prediction[0] == 1:
        st.error("The patient is likely to have diabetes. 🩸")
    else:
        st.success("The patient is not likely to have diabetes. ✅")

    st.write("### Prediction Confidence:")
    st.write(f"Probability of having diabetes: {prediction_proba[0][1]:.2%}")
    st.write(f"Probability of not having diabetes: {prediction_proba[0][0]:.2%}")

    # Feature Contribution Chart
    st.write("### Feature Contribution Chart:")
    feature_names = ["Pregnancies", "Plasma Glucose", "Diastolic BP", "Triceps Thickness", "Serum Insulin", "BMI", "Diabetes Pedigree", "Age"]
    input_values = [Pregnancies, PlasmaGlucose, DiastolicBloodPressure, TricepsThickness, SerumInsulin, BMI, DiabetesPedigree, Age]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(feature_names, input_values, color='skyblue')
    ax.set_title("Feature Contributions", fontsize=16)
    ax.set_xlabel("Value", fontsize=12)
    st.pyplot(fig)

    # Feature Explanation
    st.write("### Feature Explanation:")
    st.markdown("""
    - **Pregnancies:** Number of times the patient has been pregnant.
    - **Plasma Glucose:** Plasma glucose concentration after 2 hours in an oral glucose tolerance test.
    - **Diastolic Blood Pressure:** Diastolic blood pressure in mm Hg.
    - **Triceps Thickness:** Triceps skinfold thickness in mm.
    - **Serum Insulin:** Serum insulin concentration in µU/ml.
    - **BMI:** Body mass index (weight in kg/(height in m)^2).
    - **Diabetes Pedigree:** A function that scores the likelihood of diabetes based on family history.
    - **Age:** Patient's age in years.
    """)

# Diabetes Descriptive Chart
st.write("### Diabetes Description:")
diabetes_df = pd.DataFrame({
    "Category": ["Type 1 Diabetes", "Type 2 Diabetes", "Gestational Diabetes", "Others"],
    "Percentage": [10, 70, 15, 5]
})
fig, ax = plt.subplots()
ax.pie(diabetes_df["Percentage"], labels=diabetes_df["Category"], autopct='%1.1f%%', startangle=90, colors=['blue', 'green', 'orange', 'red'])
ax.set_title("Diabetes Types Distribution")
st.pyplot(fig)
