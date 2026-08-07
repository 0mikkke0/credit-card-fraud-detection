from pathlib import Path
import joblib
import pandas as pd

MODEL = Path(__file__).resolve().parents[1] / "models" / "fraud_model.joblib"

def predict_transaction(transaction: dict):
    bundle = joblib.load(MODEL)
    probability = float(
        bundle["pipeline"].predict_proba(pd.DataFrame([transaction]))[0, 1]
    )
    threshold = float(bundle["threshold"])
    return {
        "fraud_probability": probability,
        "prediction": "FRAUD" if probability >= threshold else "LEGITIMATE",
        "threshold": threshold,
    }
