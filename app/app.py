import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load the model
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'diamond_price_lr_log.pkl')
model = joblib.load(model_path)

# Category encodings
cut_order     = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
color_order   = ["J", "I", "H", "G", "F", "E", "D"]
clarity_order = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

# App UI
st.title("💎 Diamond Price Predictor")
st.markdown("Predict the price of a diamond based on its characteristics using a trained ML model.")

st.sidebar.header("Enter Diamond Features")

carat   = st.sidebar.number_input("Carat", min_value=0.01, max_value=5.0, value=1.0, step=0.01)
cut     = st.sidebar.selectbox("Cut", cut_order)
color   = st.sidebar.selectbox("Color", color_order)
clarity = st.sidebar.selectbox("Clarity", clarity_order)
depth   = st.sidebar.number_input("Depth %", min_value=0.0, max_value=100.0, value=61.0)
table   = st.sidebar.number_input("Table %", min_value=0.0, max_value=100.0, value=57.0)
x       = st.sidebar.number_input("Length (x)", min_value=0.0, value=5.0)
y       = st.sidebar.number_input("Width (y)", min_value=0.0, value=5.0)
z       = st.sidebar.number_input("Depth (z)", min_value=0.0, value=3.0)

# Encode categorical inputs
cut_code     = cut_order.index(cut)
color_code   = color_order.index(color)
clarity_code = clarity_order.index(clarity)

# Prepare input for prediction
input_data = pd.DataFrame({
    'carat': [carat],
    'cut': [cut_code],
    'color': [color_code],
    'clarity': [clarity_code],
    'depth': [depth],
    'table': [table],
    'x': [x],
    'y': [y],
    'z': [z]
})

# Predict log(price), then convert to original scale
log_pred = model.predict(input_data)[0]
price = np.exp(log_pred)

st.subheader("💰 Predicted Diamond Price")
st.success(f"Estimated Price: **${price:,.2f}**")

