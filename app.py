import streamlit as st
import joblib
import pandas as pd
import numpy as np
# Load the trained model
model = joblib.load('linear.sav')

st.title('Sales Prediction App')
st.write('Enter the values for TV, Radio, and Newspaper advertising budgets to predict sales.')

# Input fields for features
tv = st.number_input('TV Advertising Budget', min_value=0.0, max_value=300.0, value=100.0, step=0.1)
radio = st.number_input('Radio Advertising Budget', min_value=0.0, max_value=50.0, value=20.0, step=0.1)
newspaper = st.number_input('Newspaper Advertising Budget', min_value=0.0, max_value=120.0, value=30.0, step=0.1)

if st.button('Predict Sales'):
    # Prepare the input data as a DataFrame, including 'Unnamed: 0' with a default value
    # It's important to match the feature order and count used during training
    input_data = pd.DataFrame([{
        'Unnamed: 0': 0, # Assuming 'Unnamed: 0' is an index and can be set to 0 for new predictions
        'TV': tv,
        'Radio': radio,
        'Newspaper': newspaper
    }])

    # Make prediction
    prediction = model.predict(input_data)[0]

    st.success(f'Predicted Sales: {prediction:.2f}')
