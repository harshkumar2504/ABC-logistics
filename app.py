import streamlit as st
import joblib
import pandas as pd

# Load the trained model
logi = joblib.load('logi.sav')

st.title('Delivery Delay Prediction App')
st.write('Enter the feature values to predict delivery delay.')

# Get feature names from the global X DataFrame (assuming it's available or we can list them manually)
# For robustness, we will list them manually as X might not be available at app runtime
feature_names = [
    'Delivery_Distance',
    'Traffic_Congestion',
    'Weather_Condition',
    'Delivery_Slot',
    'Driver_Experience',
    'Num_Stops',
    'Vehicle_Age',
    'Road_Condition_Score',
    'Package_Weight',
    'Fuel_Efficiency',
    'Warehouse_Processing_Time'
]

# Create input fields for each feature
input_data = {}
for feature in feature_names:
    if 'Distance' in feature or 'Weight' in feature or 'Time' in feature or 'Efficiency' in feature:
        input_data[feature] = st.number_input(f'{feature}', value=10.0, step=0.1)
    elif 'Score' in feature or 'Age' in feature:
        input_data[feature] = st.number_input(f'{feature}', value=5, step=1)
    else:
        input_data[feature] = st.number_input(f'{feature}', value=1, step=1)

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Make prediction
if st.button('Predict Delivery Delay'):
    prediction = logi.predict(input_df)
    prediction_proba = logi.predict_proba(input_df)

    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.error('Predicted: Delivery Delay (1)')
    else:
        st.success('Predicted: No Delivery Delay (0)')
    
    st.write(f"Probability of No Delay: {prediction_proba[0][0]:.4f}")
    st.write(f"Probability of Delay: {prediction_proba[0][1]:.4f}")

st.caption('Note: 1 indicates Delay, 0 indicates No Delay.')

