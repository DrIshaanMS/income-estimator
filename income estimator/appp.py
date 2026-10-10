import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

model_path = Path(__file__).resolve().parent / "income_model.pkl"
model = joblib.load(model_path)

st.title("Income Estimator")
st.write("Enter the transaction summary to estimate income.")

deposit_count = st.number_input("Deposit count", min_value=0, value=0, step=1)
average_deposit = st.number_input("Average deposit", min_value=0.0, value=0.0, step=1.0)
deposit_std = st.number_input("Deposit standard deviation", min_value=0.0, value=0.0, step=1.0)
income_source_count = st.number_input("Income source count", min_value=0, value=0, step=1)
total_withdrawals = st.number_input("Total withdrawals", min_value=0.0, value=0.0, step=1.0)
withdrawal_count = st.number_input("Withdrawal count", min_value=0, value=0, step=1)

if st.button("Estimate income"):
    features = pd.DataFrame([{
        "deposit_count": deposit_count,
        "average_deposit": average_deposit,
        "deposit_std": deposit_std,
        "income_source_count": income_source_count,
        "total_withdrawals": total_withdrawals,
        "withdrawal_count": withdrawal_count
    }])

    try:
        prediction = model.predict(features)
        st.success(f"Estimated income: {prediction[0]:,.2f}")
    except Exception as error:
        st.error(f"Prediction failed: {error}")