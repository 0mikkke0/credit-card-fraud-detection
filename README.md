# Credit Card Fraud Detection System

End-to-end machine learning portfolio project for detecting fraudulent credit-card transactions.

## Why this project?
The classic ULB credit-card fraud benchmark contains 284,807 transactions and only 492 fraud cases (~0.172%). This makes it a strong example of highly imbalanced classification.

The project emphasizes:
- precision and recall
- F1-score
- Average Precision (PR-AUC)
- ROC-AUC
- class weighting
- threshold tuning
- reproducible ML pipelines

Dataset source: ULB Machine Learning Group / Kaggle Credit Card Fraud Detection.
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Structure
```text
credit-card-fraud-detection/
├── data/
│   └── creditcard.csv
├── models/
│   ├── fraud_model.joblib
│   └── metrics.json
├── notebooks/
│   └── fraud_detection_analysis.ipynb
├── src/
│   ├── download_data.py
│   ├── train.py
│   └── predict.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup
Recommended: Python 3.11 or 3.12.

```bash
python -m venv .venv
```

Windows CMD:
```bat
.venv\Scripts\activate.bat
pip install -r requirements.txt
python src/download_data.py
python src/train.py
streamlit run app.py
```

The downloader uses the OpenML mirror of the benchmark dataset (OpenML dataset ID 1597).

## Modeling
The training script:
1. creates stratified train/validation/test splits,
2. imputes and scales features,
3. compares Logistic Regression and Random Forest with class weighting,
4. evaluates Precision, Recall, F1, ROC-AUC and PR-AUC,
5. tunes the decision threshold on the validation set,
6. saves the best pipeline and threshold.

Accuracy is deliberately not the main metric because the fraud class is extremely rare.

## Important limitation
This is a portfolio project using an anonymized public benchmark. A real fraud system would need temporal validation, streaming infrastructure, feature stores, monitoring, concept-drift detection, latency controls, investigator feedback and business-specific cost functions.

## Interview topics
Be ready to explain class imbalance, precision vs recall, PR-AUC vs ROC-AUC, class weights, threshold tuning, false positives vs false negatives, stratified splitting, data leakage, and why real fraud systems should consider time ordering.
