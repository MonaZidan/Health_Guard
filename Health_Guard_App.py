# Import LIBs

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

# Load Saved Models

brest_cancer_model = joblib.load("brest_cancer_model_pipeline.pkl")
diabetes_model = joblib.load("diabetes_model_pipeline22.pkl")
heart_failure_model = joblib.load("Heart_model_pipeline1.pkl")

# Streamlit UI Configuration

st.set_page_config(page_title="Health Guard", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")


# Sidebar Navigator

with st.sidebar:
    selected = option_menu(None, ["About Health Guard",  "Diabetes Prediction", 'Heart Failure Prediction', "Breast Cancer Prediction"], 
        icons=['bi-info-circle',  "bi-droplet", 'bi-heart-pulse', 'bi-heart'], 
        menu_icon="cast", default_index=0,
        )
    

# Health Guard Page

if selected == "About Health Guard":
    st.title("🛡️ Welcome to Health Guard Application")
    st.write("""
    Health Guard is an AI-powered application designed to help users predict and monitor their health conditions. 
    This application provides predictions for various health conditions including breast cancer, diabetes, and heart failure. 
    By leveraging advanced machine learning models, Health Guard aims to provide accurate and reliable predictions to assist users in making informed health decisions.
    
    **Features:**
    - Breast Cancer Prediction
    - Diabetes Prediction
    - Heart Failure Prediction
    
    **How to Use:**
    - Select the desired prediction from the sidebar menu.
    - Enter the required health parameters.
    - Get instant prediction results and insights.
    
    **Disclaimer:**
    Health Guard is intended for informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
    """)

# Brest Cancer Prediction Page

if selected == "Breast Cancer Prediction":
    st.title("Breast Cancer Prediction")
    st.write("This page allows you to predict the likelihood of breast cancer based on various health parameters. ")
    st.write("Please enter the required information to get an instant prediction.")

    data =['radius_mean', 'texture_mean', 'perimeter_mean','area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean','radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se','fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst','compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']
    # User Input Data
    user_input = {}
    for feature in data:  
        user_input[feature] = st.number_input(f"Enter {feature}", format="%.4f")
    # Convert User Input 2 DataFrame
    user_input_df = pd.DataFrame(user_input, index=[0])
    # Prediction
    if st.button("Evaluate Cancer Risk"):
        prediction = brest_cancer_model.predict(user_input_df)
        if prediction == 0:
            st.success("The prediction indicates that the tumor is benign. However, please consult with a healthcare professional for a comprehensive evaluation.")
        else:
            st.error("The prediction indicates that the tumor may be malignant. Please consult with a healthcare professional for further diagnosis.")


    
# Diabetes Prediction Page

if selected == "Diabetes Prediction":
    st.title("Diabetes Prediction")
    st.image("diabetes-main.png",use_container_width=True)
    st.write("This page allows you to predict the likelihood of diabetes based on various health parameters.")
    st.write("Please enter the required information to get an instant prediction.")

    data1 = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']
    col1, co12, col3 = st.columns(3)
    # User Input Data
    with col1:
        Pregnancies = st.number_input("🤰Pregnancies Number : ", step=1)
        SkinThickness = st.number_input("📏Skin Thickness Value : ",step=1)
        DiabetesPedigreeFunction = st.number_input("🧬Diabetes Pedigree Function Value : ", format="%0.3f")

    with co12:
        Glucose = st.number_input("🩸Glucose Value : ", step=1)
        Insulin = st.number_input("💉Insulin Value : ", step=1)
        Age = st.number_input("👶Age : ", step=1)

    with col3:
        BloodPressure = st.number_input("🫀Blood Pressure Level : ", step=1)
        BMI = st.number_input("⚖️Body Mass Index Value : ", format="%0.1f")
        
    user_input = {
    'Pregnancies': Pregnancies,'Glucose': Glucose,'BloodPressure': BloodPressure,'SkinThickness': SkinThickness,
    'Insulin': Insulin,'BMI': BMI,'DiabetesPedigreeFunction': DiabetesPedigreeFunction,'Age': Age}
    user_input_df = pd.DataFrame(user_input, index=[0])

    # Prediction
    if st.button("Assess Diabetes Probability"):
        prediction = diabetes_model.predict(user_input_df)
        if prediction == 0:
            st.success("The prediction indicates that the person is not likely to have diabetes. However, maintaining a healthy lifestyle and regular check-ups are important for overall health.")
        else:
            st.error("The prediction indicates that the person may have diabetes. Please consult with a healthcare professional for further diagnosis, and consider lifestyle changes such as a balanced diet and regular exercise.")

    

# Heart Failure Prediction Page

if selected == "Heart Failure Prediction":
    st.title("Heart Failure Prediction")
    st.image("Coronary-heart-disease.jpg")
    st.write("This page allows you to predict the likelihood of heart failure based on various health parameters.")
    st.write("Please enter the required information to get an instant prediction.")

    data2 =['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS','RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
    col1, co12, col3 = st.columns(3)
    # User Input Data
    with col1:
        Age = st.number_input("👶Age : ", step=1)
        RestingBP = st.number_input("🫀Resting Blood Pressure [mm Hg] : ",step=1)
        RestingECG = st.selectbox("📊Resting Electrocardiogram : ", ["Normal","ST", "LVH"])
        Oldpeak = st.number_input("📉Oldpeak : ", format="%0.1f")

    with co12:
        Sex = st.selectbox("🚻Gender : ", ["M", "F"])
        Cholesterol = st.number_input("🩸Serum Cholesterol [mm/dl] : ", step=1)
        MaxHR = st.number_input("❤️Maximum Heart Rate Achieved : ", step=1)
        ST_Slope = st.selectbox("📈The Slope of the Peak Exercise ST Segment : ", ["Up"," Flat", "Down"])

    with col3:
        ChestPainType = st.selectbox("💔Chest Pain Type : ", ["ATA", "NAP", "ASY", "TA"])
        FastingBS = st.selectbox("🍬Fasting Blood Sugar  : ", [0, 1])
        ExerciseAngina = st.selectbox("🏃‍♂️Exercise-Induced Angina : ", ["Y", "N"])
        
    user_input = {
        'Age': Age,'Sex': Sex,'ChestPainType': ChestPainType,'RestingBP': RestingBP,"Cholesterol":Cholesterol,"FastingBS":FastingBS,
        'RestingECG': RestingECG,'MaxHR': MaxHR,'ExerciseAngina': ExerciseAngina,'Oldpeak': Oldpeak, "ST_Slope":ST_Slope}

    user_input_df = pd.DataFrame(user_input, index=[0])
    user_input_df = user_input_df[data2]

    # Display the user input data
    st.write("User Input Data:")
    st.dataframe(user_input_df)

    # Prediction
    if st.button("Predict Heart Disease Risk"):
        prediction = heart_failure_model.predict(user_input_df)
        if prediction == 0:
            st.success("The prediction indicates that the person is not likely to have heart disease. However, maintaining a healthy lifestyle and regular check-ups are important for cardiovascular health.")
        else:
            st.error("The prediction indicates that the person may have heart disease. Please consult with a healthcare professional for further diagnosis, and consider lifestyle changes such as a balanced diet, regular exercise, and stress management.")
    