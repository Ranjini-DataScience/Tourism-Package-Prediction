import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_model.joblib")
model = joblib.load(model_path)

st.title("Visit with Us - Wellness Tourism Package Predictor")
st.write("""
This application predicts the likelihood of a customer purchasing a wellness tourism package based on profile parameters. 
Enter the customer details below to generate a prediction.
""")

# Input widgets organized in columns for a clean interface
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    occupation = st.selectbox("Occupation", ['Salaried', 'Free Lancer', 'Small Business', 'Large Business'])
    gender = st.selectbox("Gender", ['Male', 'Female'])
    marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
    monthly_income = st.number_input("Monthly Income", min_value=10000, max_value=200000, value=30000)
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=120, value=15)

with col2:
    number_of_person = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    number_of_followups = st.number_input("Number of Follow-ups", min_value=1, max_value=10, value=3)
    preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
    number_of_trips = st.number_input("Number of Trips per Year", min_value=1, max_value=20, value=2)
    passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    pitch_satisfaction = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
    own_car = st.selectbox("Owns a Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    number_of_children = st.number_input("Children Visiting (< 5 years)", min_value=0, max_value=5, value=0)
    type_of_contact = st.selectbox("Type of Contact", ['Self Enquiry', 'Company Invited'])
    product_pitched = st.selectbox("Product Pitched", ['Basic', 'Standard', 'Deluxe', 'Super Deluxe', 'King'])
    designation = st.selectbox("Designation", ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP'])

# Collect inputs into a DataFrame matching model feature expectations
input_data = pd.DataFrame({
    'Age': [age],
    'CityTier': [city_tier],
    'DurationOfPitch': [duration_of_pitch],
    'NumberOfPersonVisiting': [number_of_person],
    'NumberOfFollowups': [number_of_followups],
    'PreferredPropertyStar': [preferred_property_star],
    'NumberOfTrips': [number_of_trips],
    'Passport': [passport],
    'PitchSatisfactionScore': [pitch_satisfaction],
    'OwnCar': [own_car],
    'NumberOfChildrenVisiting': [number_of_children],
    'MonthlyIncome': [monthly_income],
    'TypeofContact': [type_of_contact],
    'Occupation': [occupation],
    'Gender': [gender],
    'ProductPitched': [product_pitched],
    'MaritalStatus': [marital_status],
    'Designation': [designation]
})

# Prediction trigger
if st.button("Predict Package Purchase"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"Prediction: Customer is LIKELY to purchase the package. (Probability: {probability:.2f})")
    else:
        st.warning(f"Prediction: Customer is UNLIKELY to purchase the package. (Probability: {probability:.2f})")
