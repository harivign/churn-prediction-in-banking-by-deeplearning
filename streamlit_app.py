import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU 
import base64

def add_bg_from_local(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local(r"background_image.webp")


try:
    
    churn_model = load_model("result4.h5", custom_objects={"LeakyReLU": tf.keras.layers.LeakyReLU})
    st.write("✅ Model Loaded Successfully!")
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")

# Load encoders
import pickle
with open("encoder_go.pkl", "rb") as f:
    encoder_geography = pickle.load(f)

with open("encoder_gender.pkl", "rb") as f:
    encoder_gender = pickle.load(f)

# Load dataset
dataset = pd.read_csv("my_data.csv")

st.title("Churn Prediction in Banking")

categories = ["Geography", "Gender"]
dropdown_options = {feature: dataset[feature].unique().tolist() for feature in categories}

with st.form("Churn Prediction Form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        creditscore = st.number_input("Enter Credit Score", min_value=300, max_value=850)
    with col2:
        geography_select = st.selectbox("Geography", dropdown_options["Geography"])
        geography =int( encoder_geography.transform([[geography_select]])[0])
    with col3:
        gender_select = st.selectbox("Gender", dropdown_options["Gender"])
        gender = int(encoder_gender.transform([[gender_select]])[0])

    col4, col5, col6 = st.columns(3)
    with col4:
        age = st.number_input("Enter Age", min_value=18, max_value=100)
    with col5:
        tenure = st.number_input("Enter Tenure", min_value=0, max_value=10)
    with col6:
        balance = st.number_input("Enter Balance", min_value=0)
        
    col7, col8, col9 = st.columns(3)
    with col7:
        numofproducts = st.number_input("Enter Number of Products", min_value=1, max_value=4)
    with col8:
        hascrcard = st.selectbox("Has Credit Card?", ["No", "Yes"])
        hascrcard = 1 if hascrcard == "Yes" else 0
    with col9:
        isactivemember = st.selectbox("Is Active Member?", ["No", "Yes"])
        isactivemember = 1 if isactivemember == "Yes" else 0
    
    estimatedsalary = st.number_input("Enter Estimated Salary", min_value=0)

    submit_button = st.form_submit_button("Predict")


if submit_button:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    input_data = np.array([[creditscore, geography, gender, age, tenure, balance, numofproducts, hascrcard, isactivemember, estimatedsalary]])
    input_data1 = scaler.transform(input_data)
    prediction = churn_model.predict(input_data1)

    result_text = "Customer is **likely to churn**" if prediction[0][0] > 0.5  else "Customer is **not likely to churn**"
    st.subheader(result_text)
    st.write(f"Prediction Score: {prediction[0][0]:}")

