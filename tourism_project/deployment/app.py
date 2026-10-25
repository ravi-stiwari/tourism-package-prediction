"""
Streamlit front-end for the Wellness Tourism Package prediction model.

Loads the model committed to tourism_project/deployment/ by the pipeline,
collects user inputs into a single-row dataframe with the same columns the
model was trained on, and shows the prediction.
"""
import os
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_tourism_model.joblib")

st.set_page_config(page_title="Wellness Tourism Package Predictor", page_icon="🧳")
st.title("🧳 Wellness Tourism Package Predictor")
st.write(
    "Predict whether a customer is likely to purchase the Wellness Tourism "
    "Package before contacting them."
)


@st.cache_resource
def load_model(path=MODEL_PATH):
    return joblib.load(path)


try:
    model = load_model()
except Exception as e:  # noqa: BLE001
    st.error(
        "Could not load the trained model. Make sure the GitHub Actions "
        f"pipeline has committed best_tourism_model.joblib. Details: {e}"
    )
    st.stop()

st.header("Customer details")

col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", min_value=18, max_value=100, value=35)
    TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    DurationOfPitch = st.number_input("Duration of Pitch (min)", min_value=0.0, max_value=200.0, value=15.0)
    Occupation = st.selectbox(
        "Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"]
    )
    Gender = st.selectbox("Gender", ["Male", "Female"])
    NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=3)
    NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0.0, max_value=10.0, value=3.0)
    ProductPitched = st.selectbox(
        "Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
    )

with col2:
    PreferredPropertyStar = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    NumberOfTrips = st.number_input("Number of Trips per Year", min_value=0.0, max_value=50.0, value=2.0)
    Passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
    OwnCar = st.selectbox("Owns a Car?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    NumberOfChildrenVisiting = st.number_input("Number of Children Visiting (<5 yrs)", min_value=0.0, max_value=5.0, value=0.0)
    Designation = st.selectbox(
        "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )
    MonthlyIncome = st.number_input("Monthly Income", min_value=0.0, max_value=1000000.0, value=20000.0)

# Build a single-row dataframe matching the training feature columns
# (CustomerID and the index column were dropped during preparation).
input_df = pd.DataFrame(
    [
        {
            "Age": Age,
            "TypeofContact": TypeofContact,
            "CityTier": CityTier,
            "DurationOfPitch": DurationOfPitch,
            "Occupation": Occupation,
            "Gender": Gender,
            "NumberOfPersonVisiting": NumberOfPersonVisiting,
            "NumberOfFollowups": NumberOfFollowups,
            "ProductPitched": ProductPitched,
            "PreferredPropertyStar": PreferredPropertyStar,
            "MaritalStatus": MaritalStatus,
            "NumberOfTrips": NumberOfTrips,
            "Passport": Passport,
            "PitchSatisfactionScore": PitchSatisfactionScore,
            "OwnCar": OwnCar,
            "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
            "Designation": Designation,
            "MonthlyIncome": MonthlyIncome,
        }
    ]
)

st.subheader("Model input")
st.dataframe(input_df)

if st.button("Predict"):
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    if int(pred) == 1:
        st.success(f"Likely to PURCHASE the package (probability {proba:.2%}).")
    else:
        st.info(f"Unlikely to purchase the package (purchase probability {proba:.2%}).")
