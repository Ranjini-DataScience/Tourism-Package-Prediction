import streamlit as st
import pandas as pd
import joblib
import os

# Page Configuration
st.set_page_config(
    page_title="Visit with Us - Wellness Tourism Package Predictor",
    page_icon="✈️",
    layout="wide"
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

# --- SIDEBAR: QUICK DEMO PRESETS ---
st.sidebar.header("⚡ Quick Demo Presets")
preset = st.sidebar.selectbox(
    "Load Sample Profile:", 
    ["Custom", "High Likelihood Lead", "Low Likelihood Lead"]
)

# Adjusted preset values for distinct High vs Low classification
if preset == "High Likelihood Lead":
    p_contact, p_tier, p_occ, p_gender = 1, 0, 0, 0  # Company Invited, Tier 1, Salaried, Male
    d_age, d_dur = 38, 30.0
    p_marital, d_visiting, d_followups, p_product, p_star, d_trips = 1, 3, 2, 3, 2, 4  # Married, 3 visiting, 2 follow-ups, Super Deluxe, 5-star, 4 trips
    p_pass, d_satisfaction, p_car, d_children, p_desig, d_income = 1, 5, 1, 0, 1, 65000.0  # Yes passport, 5 satisfaction, own car, Manager, 65k income
elif preset == "Low Likelihood Lead":
    p_contact, p_tier, p_occ, p_gender = 0, 2, 3, 1  # Self Enquiry, Tier 3, Free Lancer, Female
    d_age, d_dur = 55, 5.0
    p_marital, d_visiting, d_followups, p_product, p_star, d_trips = 2, 1, 6, 0, 0, 0  # Divorced, 1 person, 6 follow-ups, Basic, 3-star, 0 trips
    p_pass, d_satisfaction, p_car, d_children, p_desig, d_income = 0, 1, 0, 2, 0, 12000.0  # No passport, 1 satisfaction, no car, 2 children, Executive, 12k income
else:  # Custom default
    p_contact, p_tier, p_occ, p_gender = 0, 0, 0, 0
    d_age, d_dur = 35, 15.0
    p_marital, d_visiting, d_followups, p_product, p_star, d_trips = 0, 2, 3, 0, 0, 2
    p_pass, d_satisfaction, p_car, d_children, p_desig, d_income = 0, 3, 0, 1, 0, 22000.0

st.sidebar.info("Select a preset above to instantly test high or low conversion profiles during your presentation!")

# --- USER INPUT FORM (3 COLUMNS) ---
st.subheader("Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=d_age)
    type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"], index=p_contact)
    city_tier = st.selectbox("City Tier", [1, 2, 3], index=p_tier)
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1.0, max_value=150.0, value=d_dur)
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"], index=p_occ)
    gender = st.selectbox("Gender", ["Male", "Female"], index=p_gender)

with col2:
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"], index=p_marital)
    num_person_visiting = st.slider("Number of Persons Visiting", 1, 5, value=d_visiting)
    num_followups = st.slider("Number of Follow-ups", 1, 6, value=d_followups)
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"], index=p_product)
    preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0], index=p_star)
    num_trips = st.number_input("Number of Trips Annually", min_value=0, max_value=30, value=d_trips)

with col3:
    passport = st.selectbox("Has Passport?", [0, 1], index=p_pass, format_func=lambda x: "Yes" if x == 1 else "No")
    pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, value=d_satisfaction)
    own_car = st.selectbox("Owns Car?", [0, 1], index=p_car, format_func=lambda x: "Yes" if x == 1 else "No")
    num_children_visiting = st.slider("Number of Children Visiting (< 5 years)", 0, 3, value=d_children)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"], index=p_desig)
    monthly_income = st.number_input("Monthly Income (INR)", min_value=1000.0, max_value=150000.0, value=d_income)

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

# Prediction Logic & UI Feedback
st.markdown("---")
if st.button("Predict Purchase Likelihood", type="primary", use_container_width=True):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    # Visual Probability Progress Bar
    st.progress(float(probability), text=f"Calculated Conversion Probability: {probability:.2%}")
    
    if prediction == 1:
        st.success(f"🎉 **High Likelihood!** This customer is likely to purchase the Wellness Tourism Package (Purchase Probability: `{probability:.2%}`). Recommended for immediate sales outreach.")
    else:
        st.info(f"💡 **Low Likelihood.** This customer is unlikely to purchase the package (Purchase Probability: `{probability:.2%}`). Campaign outreach can be deferred to conserve marketing resources.")
