import joblib
import pandas as pd
from pathlib import Path


def test_model_can_predict():
    model_path = Path(__file__).resolve().parent / "income_model.pkl"
    model = joblib.load(model_path)

    test_input = pd.DataFrame([{
        "deposit_count": 5,
        "average_deposit": 10000.0,
        "deposit_std": 2000.0,
        "income_source_count": 1,
        "total_withdrawals": 15000.0,
        "withdrawal_count": 4
    }])

    prediction = model.predict(test_input)[0]

    assert isinstance(float(prediction), float)
    assert prediction >= 0
