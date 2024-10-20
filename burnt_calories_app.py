import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Load your model
@st.cache_resource
def load_calorie_model():
    model = load_model('model.h5')
    return model

loaded_model = load_calorie_model()

# Load the scaler
scaler = joblib.load('scaler.pkl')

# Title for the app
st.title('Burnt Calories Prediction')

# User inputs
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.number_input('Age', 1, 100, 25)
height = st.number_input('Height (cm)', 100, 250, 170)
weight = st.number_input('Weight (kg)', 30, 200, 70)
duration = st.number_input('Duration (minutes)', 1, 180, 60)
heart_rate = st.number_input('Heart Rate', 60, 200, 100)
body_temp = st.number_input('Body Temperature (Celsius)', 35.0, 43.0, 37.0)

# Create user input array
user_inputs = np.array([[1 if gender == 'Male' else 0, age, height, weight, duration, heart_rate, body_temp]])

# Scale the user inputs
user_inputs_scaled = scaler.transform(user_inputs)

if st.button('Predict'):
    prediction = loaded_model.predict(user_inputs_scaled)
    st.markdown(f'**The predicted burnt calories is: {prediction[0][0]:,.2f}**')

    with st.expander("Show more details"):
        st.write("Details of the prediction:")
        st.write('Model used: Sequential Neural Network')

