import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("bean_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page config
st.set_page_config(page_title="Dry Bean Classifier", layout="wide")

# Title
st.title("🌱 Dry Bean Classification App")
st.markdown("Predict the type of dry bean based on physical characteristics")

# Sidebar inputs
st.sidebar.header("Enter Bean Features")

def user_input():
    Area = st.sidebar.number_input("Area", value=10000.0)
    Perimeter = st.sidebar.number_input("Perimeter", value=300.0)
    MajorAxisLength = st.sidebar.number_input("Major Axis Length", value=200.0)
    MinorAxisLength = st.sidebar.number_input("Minor Axis Length", value=100.0)
    AspectRatio = st.sidebar.number_input("Aspect Ratio", value=1.5)
    Eccentricity = st.sidebar.number_input("Eccentricity", value=0.8)
    ConvexArea = st.sidebar.number_input("Convex Area", value=10500.0)
    EquivDiameter = st.sidebar.number_input("Equivalent Diameter", value=120.0)
    Extent = st.sidebar.number_input("Extent", value=0.7)
    Solidity = st.sidebar.number_input("Solidity", value=0.95)
    Roundness = st.sidebar.number_input("Roundness", value=0.85)
    Compactness = st.sidebar.number_input("Compactness", value=0.8)
    ShapeFactor1 = st.sidebar.number_input("ShapeFactor1", value=0.003)
    ShapeFactor2 = st.sidebar.number_input("ShapeFactor2", value=0.001)
    ShapeFactor3 = st.sidebar.number_input("ShapeFactor3", value=0.8)
    ShapeFactor4 = st.sidebar.number_input("ShapeFactor4", value=0.99)

    data = np.array([[Area, Perimeter, MajorAxisLength, MinorAxisLength,
                      AspectRatio, Eccentricity, ConvexArea, EquivDiameter,
                      Extent, Solidity, Roundness, Compactness,
                      ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4]])

    return data

# Get input
input_data = user_input()

# Scale input
input_scaled = scaler.transform(input_data)

# Prediction button
if st.button("Predict"):
    prediction = model.predict(input_scaled)
    
    st.subheader("Prediction Result")
    st.success(f"🌾 Predicted Bean Type: **{prediction[0]}**")

# Optional display
if st.checkbox("Show Input Data"):
    st.write(input_data)
