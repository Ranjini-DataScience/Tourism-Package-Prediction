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
    age_input = st.text_input("Age", value="35")
    occupation = st.selectbox("Occupation", ['Salaried', 'Free Lancer', 'Small Business', 'Large Business'])
    gender = st.selectbox("Gender", ['Male', 'Female'])
    marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
    income_input = st.text_input("Monthly Income", value="30000")
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=120, value=15)

with col2:
    persons_input = st.text_input("Number of Persons Visiting", value="2")
    followups_input = st.text_input("Number of Follow-ups", value="3")
    preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
    trips_input = st.text_input("Number of Trips per Year", value="2")
    passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    pitch_satisfaction = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5], index=2)
    own_car = st.selectbox("Owns a Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    children_input = st.text_input("Children Visiting (< 5 years)", value="0")
    type_of_contact = st.selectbox("Type of Contact", ['Self Enquiry', 'Company Invited'])
    product_pitched = st.selectbox("Product Pitched", ['Basic', 'Standard', 'Deluxe', 'Super Deluxe', 'King'])
    designation = st.selectbox("Designation", ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP'])

# Safe type conversions for model compatibility
try:
    age = int(age_input)
except ValueError:
    age = 35

try:
    monthly_income = float(income_input)
except ValueError:
    monthly_income = 30000.0

try:
    number_of_person = int(persons_input)
except ValueError:
    number_of_person = 2

try:
    number_of_followups = int(followups_input)
except ValueError:
    number_of_followups = 3

try:
    number_of_trips = int(trips_input)
except ValueError:
    number_of_trips = 2

try:
    number_of_children = int(children_input)
except ValueError:
    number_of_children = 0

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

# Prediction trigger button and output display
if st.button("Predict Package Purchase"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"Prediction: Customer is LIKELY to purchase the package. (Probability: {probability:.2f})")
    else:
        st.warning(f"Prediction: Customer is UNLIKELY to purchase the package. (Probability: {probability:.2f})")
