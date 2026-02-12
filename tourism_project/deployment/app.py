import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
# Updated repo_id and filename for Wellness Tourism model
model_repo_id = "ShrutiHulyal/WellnessTourismPackagePurchasemodel"
model_filename = "best_wellness_tourism_model_v1.joblib"
model_path = hf_hub_download(repo_id=model_repo_id, filename=model_filename)
model = joblib.load(model_path)

# Streamlit UI for Wellness Tourism Package Purchase Prediction
st.title("Wellness Tourism Package Purchase Prediction App")
st.write("""
This application predicts whether a customer will purchase the newly introduced Wellness Tourism Package
before contacting them. Please enter customer and interaction details below to get a prediction.
""")

# User input fields
st.header("Customer Details")
age = st.number_input("Age", min_value=18, max_value=90, value=30)

type_of_contact_options = {'Company Invited': 0, 'Self Inquiry': 1}
type_of_contact_selected = st.selectbox("Type of Contact", options=list(type_of_contact_options.keys()))
type_of_contact_encoded = type_of_contact_options[type_of_contact_selected]

city_tier = st.selectbox("City Tier", options=[1, 2, 3], index=0) # Assuming 1, 2, 3 mapping

# Mappings for categorical features based on training data
occupation_mapping = {'Salaried': 0, 'Free Lancer': 1, 'Small Business': 2, 'Large Business': 3}
occupation_selected = st.selectbox("Occupation", options=list(occupation_mapping.keys()))
occupation_encoded = occupation_mapping[occupation_selected]

gender_mapping = {'Female': 0, 'Male': 1}
gender_selected = st.selectbox("Gender", options=list(gender_mapping.keys()))
gender_encoded = gender_mapping[gender_selected]

num_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
preferred_property_star = st.number_input("Preferred Property Star (1-5)", min_value=1, max_value=5, value=3)

marital_status_mapping = {'Single': 0, 'Divorced': 1, 'Married': 2, 'Unmarried': 3}
marital_status_selected = st.selectbox("Marital Status", options=list(marital_status_mapping.keys()))
marital_status_encoded = marital_status_mapping[marital_status_selected]

num_of_trips = st.number_input("Number of Trips Annually", min_value=0, max_value=50, value=5)
passport = st.selectbox("Holds Passport?", options=[0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
own_car = st.selectbox("Owns Car?", options=[0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
num_children_visiting = st.number_input("Number of Children Visiting (below age 5)", min_value=0, max_value=5, value=0)

designation_mapping = {'Manager': 0, 'Executive': 1, 'Senior Manager': 2, 'AVP': 3, 'VP': 4}
designation_selected = st.selectbox("Designation", options=list(designation_mapping.keys()))
designation_encoded = designation_mapping[designation_selected]

monthly_income = st.number_input("Monthly Income", min_value=0.0, max_value=1000000.0, value=50000.0, step=1000.0)

st.header("Customer Interaction Data")
pitch_satisfaction_score = st.number_input("Pitch Satisfaction Score (1-5)", min_value=1, max_value=5, value=3)

product_pitched_mapping = {'Deluxe': 0, 'Basic': 1, 'Standard': 2, 'Super Deluxe': 3, 'King': 4}
product_pitched_selected = st.selectbox("Product Pitched", options=list(product_pitched_mapping.keys()))
product_pitched_encoded = product_pitched_mapping[product_pitched_selected]

num_of_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=2)
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=120, value=15)

# Assemble input into DataFrame, ensuring correct column order and names as per Xtrain
# The order of columns here should match the Xtrain dataframe *before* preprocessing by the pipeline
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': type_of_contact_encoded,
    'CityTier': city_tier,
    'Occupation': occupation_encoded,
    'Gender': gender_encoded,
    'NumberOfPersonVisiting': num_person_visiting,
    'PreferredPropertyStar': preferred_property_star,
    'MaritalStatus': marital_status_encoded,
    'NumberOfTrips': num_of_trips,
    'Passport': passport,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': num_children_visiting,
    'Designation': designation_encoded,
    'MonthlyIncome': monthly_income,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'ProductPitched': product_pitched_encoded,
    'NumberOfFollowups': num_of_followups,
    'DurationOfPitch': duration_of_pitch
}])


if st.button("Predict Purchase"):
    # Predict probability
    prediction_proba = model.predict_proba(input_data)[:, 1]

    # Use the classification threshold from training (0.45)
    classification_threshold = 0.45
    prediction = (prediction_proba >= classification_threshold).astype(int)[0]

    result = "Will Purchase the Wellness Tourism Package" if prediction == 1 else "Will Not Purchase the Wellness Tourism Package"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}** (Probability: {prediction_proba[0]:.2f})")
