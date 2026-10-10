# Bank Transaction Income Estimator

A machine-learning demo that estimates income from banking transaction summary features.

## How it works

The Streamlit app collects six summary inputs and passes them to a trained scikit-learn regression model.

Inputs:
- Deposit count
- Average deposit
- Deposit standard deviation
- Number of income sources
- Total withdrawals
- Withdrawal count

## Run locally

Install the packages:

```bash
pip install -r "income estimator/requirements.txt"
```

Run the app from the repository folder:

```bash
streamlit run "income estimator/app1.py"
```

## Important limitation

This demo asks users to enter transaction summaries manually. It does not connect to a bank or calculate summaries from raw transactions. Predictions are estimates, not financial advice.
