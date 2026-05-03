"""Streamlit frontend for the Wellness Tourism purchase predictor."""

import os

import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

# MUST be the first Streamlit call
st.set_page_config(page_title="Wellness Tourism Predictor", page_icon=":palm_tree:")

HF_USERNAME = os.getenv("HF_USERNAME", "mervml")
MODEL_REPO = f"{HF_USERNAME}/wellness-tourism-model"
MODEL_FILE = "best_wellness_tourism_model_v1.joblib"


@st.cache_resource
def load_model():
    path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    return joblib.load(path)


model = load_model()

st.title("Visit With Us - Wellness Tourism Package Predictor")
st.write(
    """
    This app predicts whether a customer is likely to purchase the new
    **Wellness Tourism Package**. Enter the customer's profile and the
    most recent sales-pitch interaction details below to get a prediction.
    """
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer profile")
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation",
        ["Salaried", "Free Lancer", "Small Business", "Large Business"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    designation = st.selectbox("Designation",
        ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input(
        "Monthly Income (in local currency)",
        min_value=0.0, max_value=200000.0, value=22000.0, step=500.0,
    )
    passport = st.selectbox("Has Passport?", [0, 1])
    own_car = st.selectbox("Owns Car?", [0, 1])

with col2:
    st.subheader("Trip & sales-pitch details")
    number_of_trips = st.number_input("Average Number of Trips per Year",
                                      min_value=0, max_value=30, value=3)
    number_of_persons = st.number_input("Number of Persons Visiting",
                                        min_value=1, max_value=10, value=3)
    number_of_children = st.number_input("Number of Children Visiting (under 5)",
                                         min_value=0, max_value=5, value=1)
    preferred_property_star = st.selectbox("Preferred Property Star Rating",
                                           [3.0, 4.0, 5.0])
    product_pitched = st.selectbox("Product Pitched",
        ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)",
                                        min_value=1.0, max_value=60.0,
                                        value=15.0, step=1.0)
    number_of_followups = st.number_input("Number of Follow-ups",
                                          min_value=0, max_value=10, value=3)
    pitch_satisfaction_score = st.selectbox("Pitch Satisfaction Score",
                                            [1, 2, 3, 4, 5])

input_df = pd.DataFrame(
    [{
        "Age": age, "TypeofContact": type_of_contact, "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch, "Occupation": occupation,
        "Gender": gender, "NumberOfPersonVisiting": number_of_persons,
        "NumberOfFollowups": number_of_followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": preferred_property_star,
        "MaritalStatus": marital_status, "NumberOfTrips": number_of_trips,
        "Passport": passport, "PitchSatisfactionScore": pitch_satisfaction_score,
        "OwnCar": own_car, "NumberOfChildrenVisiting": number_of_children,
        "Designation": designation, "MonthlyIncome": monthly_income,
    }]
)

if st.button("Predict purchase likelihood"):
    proba = model.predict_proba(input_df)[0, 1]
    pred = int(proba >= 0.5)
    st.subheader("Prediction")
    if pred == 1:
        st.success(
            f"Likely to purchase the Wellness Tourism Package (probability {proba:.2%})."
        )
    else:
        st.warning(
            f"Unlikely to purchase the Wellness Tourism Package (probability {proba:.2%})."
        )
    st.caption("Threshold: 0.5 - tune in the model serving layer if needed.")
