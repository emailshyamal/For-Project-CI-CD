import os
import streamlit as st
import pandas as pd
import joblib

# Load the tourism holiday package prediction model
model_path = os.path.join(os.path.dirname(__file__), "package_prediction.joblib")
model = joblib.load(model_path)

st.title("Holiday Package Purchase Prediction App")
st.write("""
This application predicts whether a customer will purchase a holiday package based on their profile data.
Enter the customer details below to get a prediction.
""")

# Categorical Inputs
typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
occupation     = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
gender         = st.selectbox("Gender", ["Male", "Female"])
product_pitched= st.selectbox("Product Pitched", ["Deluxe", "Standard", "Basic", "Super Deluxe", "King"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
designation    = st.selectbox("Designation", ["Manager", "Executive", "Senior Executive", "AVP", "VP"])

# Numeric Inputs
age                        = st.number_input("Age", 18.0, 100.0, 35.0, 1.0)
city_tier                  = st.selectbox("City Tier", [1, 2, 3], index=0)
duration_of_pitch          = st.number_input("Duration of Pitch (mins)", 0.0, 120.0, 15.0, 1.0)
number_of_person_visiting  = st.number_input("Number of Persons Visiting", 1, 10, 3)
number_of_followups        = st.number_input("Number of Follow-ups", 0.0, 10.0, 3.0, 1.0)
preferred_property_star    = st.selectbox("Preferred Property Star Rating", [3.0, 4.0, 5.0], index=0)
number_of_trips            = st.number_input("Number of Trips", 0.0, 20.0, 3.0, 1.0)
passport                   = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
pitch_satisfaction_score   = st.slider("Pitch Satisfaction Score", 1, 5, 3)
own_car                    = st.selectbox("Owns a Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
number_of_children_visiting= st.number_input("Number of Children Visiting", 0, 5, 1)
monthly_income             = st.number_input("Monthly Income", 0.0, 500000.0, 250000.0, 1000.0)

# Match exact DataFrame schema used during pipeline training
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": typeof_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": number_of_person_visiting,
    "NumberOfFollowups": number_of_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_of_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": number_of_children_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
}])

if st.button("Predict Purchase"):
    # Using predict_proba if custom thresholding (e.g., 0.45) is needed, otherwise predict()
    try:
        # Check if predict_proba can be safely applied for custom classification thresholding
        probabilities = model.predict_proba(input_data)[:, 1][0]
        prediction = 1 if probabilities >= 0.45 else 0
    except (AttributeError, IndexError):
        prediction = model.predict(input_data)[0]
        
    result = "Will Take Holiday Package" if prediction == 1 else "Will Not Take Holiday Package"
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"The model predicts: **{result}**")
    else:
        st.info(f"The model predicts: **{result}**")
