import streamlit as st
import pandas as pd
import joblib
import os

# Page Configuration
st.set_page_config(
    page_title="Visit with Us - Wellness Tourism Package Predictor",
    page_icon="✈️",
    layout="wide"  # Changed to "wide" layout to accommodate 3 columns smoothly!
)

# App Header with Company Name and Enhanced Title
st.title("✈️ Visit with Us: Wellness Tourism Package Predictor")
st.markdown("### Lead Conversion & Campaign Intelligence Portal")
st.write(
    "Welcome to the **Visit with Us** predictive intelligence platform. "
    "This application analyzes customer profile parameters to predict the likelihood of purchasing our newly introduced Wellness Tourism Package, optimizing lead targeting and sales outreach."
)

# Robust path resolution to locate the model file in the same directory as app.py
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "best_tourism_model.joblib")

@st.cache_resource
def load_model():
    return joblib.load(model_path)

model = load_model()

# User Input Form
st.subheader("Customer Details")

# Creating 3 columns for a compact, single-page layout
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1.0, max_value=150.0, value=15.0)
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])

with col2:
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    num_person_visiting = st.slider("Number of Persons Visiting", 1, 5, 2)
    num_followups = st.slider("Number of Follow-ups", 1, 6, 3)
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
    preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    num_trips = st.number_input("Number of Trips Annually", min_value=0, max_value=30, value=2)

with col3:
    passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    own_car = st.selectbox("Owns Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    num_children_visiting = st.slider("Number of Children Visiting (< 5 years)", 0, 3, 1)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income (INR)", min_value=1000.0, max_value=150000.0, value=22000.0)

# Compile Inputs into DataFrame
input_data = pd.DataFrame({
    'Age': [age],
    'TypeofContact': [type_of_contact],
    'CityTier': [city_tier],
    'DurationOfPitch': [duration_of_pitch],
    'Occupation': [occupation],
    'Gender': [gender],
    'NumberOfPersonVisiting': [num_person_visiting],
    'NumberOfFollowups': [num_followups],
    'ProductPitched': [product_pitched],
    'PreferredPropertyStar': [preferred_property_star],
    'MaritalStatus': [marital_status],
    'NumberOfTrips': [num_trips],
    'Passport': [passport],
    'PitchSatisfactionScore': [pitch_satisfaction_score],
    'OwnCar': [own_car],
    'NumberOfChildrenVisiting': [num_children_visiting],
    'Designation': [designation],
    'MonthlyIncome': [monthly_income]
})

# Prediction Logic
st.markdown("---")
if st.button("Predict Purchase Likelihood", type="primary", use_container_width=True):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    if prediction == 1:
        st.success(f"🎉 **High Likelihood!** This customer is likely to purchase the Wellness Tourism Package (Purchase Probability: `{probability:.2%}`). Recommended for immediate sales outreach.")
    else:
        st.info(f"💡 **Low Likelihood.** This customer is unlikely to purchase the package (Purchase Probability: `{probability:.2%}`). Campaign outreach can be deferred to conserve marketing resources.")
