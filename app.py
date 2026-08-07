from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "fraud_model.joblib"

st.set_page_config(page_title="Fraud Detection", page_icon="💳")
st.title("💳 Credit Card Fraud Detection")
st.caption("Portfolio ML application using an anonymized public benchmark.")

if not MODEL.exists():
    st.error("Model not found. Run `python src/train.py` first.")
    st.stop()

bundle = joblib.load(MODEL)
model = bundle["pipeline"]
threshold = float(bundle["threshold"])

st.info("V1–V28 are anonymized PCA-derived features. Enter transaction values for a demo prediction.")

values = {}
cols = st.columns(3)
for i in range(1, 29):
    with cols[(i - 1) % 3]:
        values[f"V{i}"] = st.number_input(f"V{i}", value=0.0, format="%.6f")

values["Time"] = st.number_input("Time", min_value=0.0, value=0.0)
values["Amount"] = st.number_input("Amount", min_value=0.0, value=50.0)

if st.button("Analyze Transaction", type="primary"):
    probability = float(model.predict_proba(pd.DataFrame([values]))[0, 1])
    st.metric("Fraud probability", f"{probability:.2%}")
    st.caption(f"Decision threshold: {threshold:.2f}")

    if probability >= threshold:
        st.error("⚠️ Transaction flagged for review")
    else:
        st.success("✅ Transaction classified as legitimate")
