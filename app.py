import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("fraud_model.pkl", "rb"))

st.title("💳 Credit Card Fraud Detection (Linear Regression)")

st.write("Enter transaction details to estimate fraud likelihood:")

# Input fields
amount = st.number_input("Transaction Amount (₹)", 10.0, 5000.0, 100.0)
time = st.number_input("Transaction Time (0-23 hrs)", 0, 23, 10)
location_match = st.selectbox("Location Matches User Profile?", ["Yes", "No"])
swipe_type = st.selectbox("Card Swipe Type", ["Offline Swipe", "Chip Payment", "Online Payment"])
past_history = st.number_input("Past Fraud Count", 0, 10, 0)

# Convert inputs
loc = 1 if location_match == "Yes" else 0
swipe_map = {"Offline Swipe": 0, "Chip Payment": 1, "Online Payment": 2}
swipe_val = swipe_map[swipe_type]

if st.button("Predict Fraud Risk"):
    input_data = np.array([[amount, time, loc, swipe_val, past_history]])
    prediction = model.predict(input_data)[0]

    # Clip prediction between 0 and 1
    risk = max(0, min(1, prediction))

    st.subheader("🟢 Prediction Result:")
    st.write(f"Fraud Likelihood Score: **{risk:.2f}** (0 = Safe, 1 = High Risk)")

    if risk > 0.7:
        st.error("⚠ High Fraud Risk Detected!")
    elif risk > 0.3:
        st.warning("⚠ Moderate Fraud Risk! Need manual verification.")
    else:
        st.success("🟢 Low Fraud Risk.")
