import streamlit as st 
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.models import Sequential

def load_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)

churn_model = load_model('result.h5')
encoder_geography = load_model('encoder_geography.pkl')
encoder_gender = load_model('encoder_gender.pkl')

# Load dataset
dataset = pd.read_csv('cleaned_churn_data.csv')

st.title('Churn Prediction in Banking')
st.write('Customer churn in banking refers to the likelihood of a customer leaving the bank (e.g., closing their account). Churn prediction uses machine learning to analyze customer behavior and identify those at risk of leaving.')

categories = ["Geography", "Gender"]
dropdown_options = {feature: dataset[feature].unique().tolist() for feature in categories}

with st.form("Churn Prediction Form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        creditscore = st.number_input("Enter Credit Score", min_value=300, max_value=850)
    with col2:
        geography_select = st.selectbox('Geography', dropdown_options['Geography'])
        geography = encoder_geography.transform([[geography_select]])[0]  # Fixed
    with col3:
        gender_select = st.selectbox('Gender', dropdown_options['Gender'])
        gender = encoder_gender.transform([[gender_select]])[0]  # Fixed

    col4, col5, col6 = st.columns(3)
    with col4:  # Fixed column misalignment
        age = st.number_input("Enter Age", min_value=18, max_value=100)
    with col5:
        tenure = st.number_input("Enter Tenure", min_value=0, max_value=10)
    with col6:
        balance = st.number_input("Enter Balance", min_value=0)

    col7, col8, col9 = st.columns(3)
    with col7:
        numofproducts = st.number_input("Enter Number of Products", min_value=1, max_value=4)
    with col8:
        hascrcard = st.number_input("Has Credit Card? (0 = No, 1 = Yes)", min_value=0, max_value=1)
    with col9:
        isactivemember = st.number_input("Is Active Member? (0 = No, 1 = Yes)", min_value=0, max_value=1)

    col10 = st.columns(1)
    with col10[0]:  # Corrected single column usage
        estimatedsalary = st.number_input("Enter Estimated Salary", min_value=0)

    # Prediction on button click
    if st.form_submit_button(label='Predict'):
        input_data = np.array([[creditscore, geography, gender, age, tenure, balance, numofproducts, hascrcard, isactivemember, estimatedsalary]])
        prediction = churn_model.predict(input_data)
        
        result_text = "Customer is **likely to churn**" if prediction[0] == 1 else "Customer is **not likely to churn**"
        st.subheader(result_text)

