import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model and scaler
model = joblib.load('knn_model.pkl')  
scaler = joblib.load('scaler.pkl')  

# Define the prediction function
def predict_quality(input_data):
    # Scale the input data
    scaled_data = scaler.transform(input_data)
    # Make the prediction
    prediction = model.predict(scaled_data)
    return prediction

# Streamlit app title
st.title("Wine Quality Prediction")

# Input fields for wine features
fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0)
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0)
citric_acid = st.number_input("Citric Acid", min_value=0.0)
residual_sugar = st.number_input("Residual Sugar", min_value=0.0)
chlorides = st.number_input("Chlorides", min_value=0.0)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0)
density = st.number_input("Density", min_value=0.0)
pH = st.number_input("pH", min_value=0.0)
sulphates = st.number_input("Sulphates", min_value=0.0)
alcohol = st.number_input("Alcohol", min_value=0.0)

# Create a DataFrame from the input values
input_df = pd.DataFrame([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, 
                         free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]],
                       columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
                                'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 
                                'pH', 'sulphates', 'alcohol'])

# Make prediction when the button is clicked
if st.button("Predict"):
    prediction = predict_quality(input_df)
    if prediction[0] == 1:
        st.success("This wine is predicted to be of Good Quality!")
    else:
        st.error("This wine is predicted to be of Bad Quality.")