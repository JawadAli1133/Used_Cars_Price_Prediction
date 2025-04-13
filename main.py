import streamlit as st
import pandas as pd
import joblib

rf_model = joblib.load('random_forest_model.pkl')
label_encoders = joblib.load('label_encoder.pkl')
cars_df = pd.read_csv('preowned_cars.csv')

st.set_page_config(page_title="Used Car Price Predictor", layout="centered")

st.title("🚗 Used Car Price Predictor")
st.markdown("Enter the car details below to predict the **estimated price**.")

# 1. Brand
brands = sorted(cars_df['brand'].dropna().unique())
brand = st.selectbox("Select Brand", brands)

# 2. Model - filtered by brand
filtered_models = sorted(cars_df[cars_df['brand'] == brand]['model'].dropna().unique())
model = st.selectbox("Select Model", filtered_models)

# 3. Transmission
transmission = st.selectbox("Transmission Type", ['Manual', 'Automatic'])

# 4. Make Year
make_year = st.number_input("Enter Make Year", min_value=2000, max_value=2024, step=1)

# 5. Fuel Type
fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])

# 6. Engine Capacity
engine_capacity = st.number_input("Enter Engine Capacity (cc)", min_value=500, max_value=5000, step=100)

# 6.1 KM Driven
km_driven = st.number_input("Enter KM Driven", min_value=0, max_value=500000, step=500)

# 7. Ownership
ownership = st.selectbox("Ownership", ['1st owner', '2nd owner', '3rd owner', '4th owner', 'Other'])

# 8. Overall Cost
overall_cost = st.number_input("Enter Overall Cost (in ₹)", min_value=10000, max_value=5000000, step=1000)

# 9. Insurance
has_insurance = st.selectbox("Has Insurance?", ['Yes', 'No'])

# 10. Spare Key
spare_key = st.selectbox("Spare Key Available?", ['Yes', 'No'])

# 11. Registration Year

reg_year_only = st.number_input("Enter Registration Year", min_value=2000, max_value=2024, step=1)

# --- Process and Predict ---
if st.button("Predict Price"):

    # Mapping Yes/No to 1/0
    has_insurance = 1 if has_insurance == 'Yes' else 0
    spare_key = 1 if spare_key == 'Yes' else 0

    # Mapping ownership
    ownership_map = {
        '1st owner': 1,
        '2nd owner': 2,
        '3rd owner': 3,
        '4th owner': 4,
        'Other': 5
    }
    ownership_encoded = ownership_map.get(ownership, 5)

    # Encode categorical using loaded label encoders
    brand_encoded = label_encoders['brand'].transform([brand])[0]
    model_encoded = label_encoders['model'].transform([model])[0]
    fuel_encoded = label_encoders['fuel_type'].transform([fuel_type])[0]
    trans_encoded = label_encoders['transmission'].transform([transmission])[0]

    # Create final input DataFrame
    input_data = pd.DataFrame([[
        brand_encoded,
        model_encoded,
        trans_encoded,
        make_year,
        fuel_encoded,
        engine_capacity,
        km_driven,
        ownership_encoded,
        overall_cost,
        has_insurance,
        spare_key,
        reg_year_only
    ]], columns=[
        'brand', 'model', 'transmission', 'make_year', 'fuel_type',
        'engine_capacity(CC)', 'km_driven',
        'ownership', 'overall_cost',
        'has_insurance', 'spare_key', 'reg_year_only'
    ])

    # Prediction
    prediction = rf_model.predict(input_data)[0]
    st.success(f"💰 Predicted Car Price: ₹{int(prediction):,}")