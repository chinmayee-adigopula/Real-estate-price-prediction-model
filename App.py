import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
import streamlit_lottie as st_lottie
import json

# Set wide layout
st.set_page_config(page_title="🏡 House Price Prediction", layout="centered")

# CSS to add GIF background and brighter fonts
st.markdown("""
    <style>
    body {
        background: url("https://s5.ezgif.com/tmp/ezgif-5d93ea55a0b53d.gif");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: white;
    }
    .stApp {
        background: rgba(0, 0, 0, 0.7);
        color: white;
    }
    .gradient-text {
        background: linear-gradient(90deg, #00f5d4, #00b4d8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtext {
        text-align: center;
        color: #f5f5f5;
        font-size: 18px;
        margin-bottom: 20px;
    }
    .stSidebar {
        background-color: rgba(17, 17, 17, 0.9);
        color: #ff4d6d; /* Reddish Pink */
    }
    .stSidebar select, .stSidebar input, .stSidebar textarea {
        color: black !important;
        background-color: #333 !important;
    }
    .stSidebar label, .stSidebar div, .stSidebar * {
        color: #ff4d6d !important;
    }
    .stButton>button {
        color: black !important;
    }
    div[data-testid="stLottie"] {
        background-color: black;
        padding: 20px;
        border-radius: 12px;
    }
    </style>
""", unsafe_allow_html=True)



# Gradient header
st.markdown('<div class="gradient-text">🏡 Bengaluru House Price Prediction</div>', unsafe_allow_html=True)
st.write("---")

# Lottie animation
with open("house_animation.json", "r") as f:
    animation = json.load(f)
st_lottie.st_lottie(animation, height=200, key="house")

# Load model and columns
model = joblib.load('GradientBoosting_model.pkl')
feature_columns = joblib.load('model_columns.pkl')
locations = [col for col in feature_columns if col not in ['total_sqft', 'bath', 'BHK']]

# Sidebar inputs
st.sidebar.header("🏠 Property Details")
location = st.sidebar.selectbox('📍 Location', locations)
total_sqft = st.sidebar.slider('📏 Total Square Feet', 1, 52272, 100)
bath = st.sidebar.slider('🚿 Bathroom', 1, 18, 1)
bhk = st.sidebar.slider('🛏️ BHK', 1, 18, 1)

# Feature vector
input_dict = {'total_sqft': total_sqft, 'bath': bath, 'BHK': bhk}
for loc in locations:
    input_dict[loc] = 1 if loc == location else 0
input_df = pd.DataFrame([input_dict])[feature_columns]

# Prediction button
if st.button('🔮 Predict Price'):
    with st.spinner('Crunching the numbers... ⏳'):
        time.sleep(1.5)
        prediction = model.predict(input_df)
    st.success(f'🏷️ Estimated Price: ₹ {np.round(prediction[0], 2)}')
    st.balloons()