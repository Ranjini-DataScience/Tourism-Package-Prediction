import streamlit as st
import pandas as pd
import joblib
import os

# Page Configuration
st.set_page_config(
    page_title="Visit with Us - Wellness Tourism Package Predictor",
    page_icon="✈️",
    layout="centered"
)

# App Header with Company Name and Enhanced Title
st.title("✈️ Visit with Us: Wellness Tourism Package Predictor")
st.markdown("### Lead Conversion & Campaign Intelligence Portal")
st.write(
    "Welcome to the **Visit with Us** predictive intelligence platform. "
    "This application analyzes customer profile parameters to predict the likelihood of purchasing our newly introduced Wellness Tourism Package, optimizing lead targeting and sales outreach."
)

# Load the model committed by the pipeline (looks next to this file)
model_path = "best_model.joblib" # Update filename if yours differs
model = joblib.load(model_path)

# ... (keep the rest of your input widgets, data collection, and prediction code below as it is)
