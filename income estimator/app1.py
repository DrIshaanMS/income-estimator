import streamlit as st
import joblib
import pandas as pd
from pathlib import Path


st.set_page_config(
    page_title="Income Estimator",
    page_icon="💰",
    layout="centered"
)


@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent / "income_model.pkl"
    return joblib.load(model_path)


model = load_model()

st.title("💰 Income Estimator")
st.caption("Estimate income from a summary of banking transactions.")
st.divider()

st.subheader("Transaction summary")
st.write("Enter figures for the same period used to prepare your model data.")

with st.form("income_form"):
    col1, col2 = st.columns(2)

    with col1:
        deposit_count = st.number_input(
            "Number of deposits",
            min_value=0,
            value=0,
            step=1
        )
        average_deposit = st.number_input(
            "Average deposit",
            min_value=0.0,
            value=0.0,
            step=100.0
        )
        deposit_std = st.number_input(
            "Deposit variability",
            min_value=0.0,
            value=0.0,
            step=100.0,
            help="Standard deviation of deposit amounts."
        )

    with col2:
        income_source_count = st.number_input(
            "Number of income sources",
            min_value=0,
            value=0,
            step=1
        )
        total_withdrawals = st.number_input(
            "Total withdrawals",
            min_value=0.0,
            value=0.0,
            step=100.0
        )
        withdrawal_count = st.number_input(
            "Number of withdrawals",
            min_value=0,
            value=0,
            step=1
        )

    submitted = st.form_submit_button(
        "Estimate income",
        type="primary",
        use_container_width=True
    )

if submitted:
    features = pd.DataFrame([{
        "deposit_count": deposit_count,
        "average_deposit": average_deposit,
        "deposit_std": deposit_std,
        "income_source_count": income_source_count,
        "total_withdrawals": total_withdrawals,
        "withdrawal_count": withdrawal_count
    }])

    try:
        prediction = float(model.predict(features)[0])

        st.divider()
        st.subheader("Estimated income")
        st.metric("Model estimate", f"{prediction:,.2f}")
        st.caption(
            "The estimate uses the same units as the income values "
            "used to train the model."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

st.divider()
st.caption("This is a model estimate, not a guarantee of actual income.")