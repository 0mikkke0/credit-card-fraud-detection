# Credit Card Fraud Detection System

An end-to-end machine learning project for detecting fraudulent credit-card transactions, with a focus on highly imbalanced classification.

## Why This Project?

The benchmark dataset contains 284,807 transactions, of which only 492 are fraudulent (~0.172%). This makes it a strong example of highly imbalanced classification, where accuracy alone can be misleading.

The project focuses on:

* Precision and Recall
* F1-score
* Average Precision (PR-AUC)
* ROC-AUC
* Class weighting
* Decision-threshold tuning
* Stratified data splitting
* Reproducible scikit-learn pipelines

## Dataset

This project uses the Credit Card Fraud Detection benchmark associated with the ULB Machine Learning Group and distributed through Kaggle.

The dataset is downloaded automatically using the project's data-download script, so the large raw dataset does not need to be committed to GitHub.

## Project Structure

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

Create and activate a virtual environment:

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate.bat
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Download the dataset:

```bash
python src/download_data.py
```

Train and evaluate the models:

```bash
python src/train.py
```

Launch the Streamlit application:

```bash
streamlit run app.py
```

The downloader uses an OpenML mirror of the benchmark dataset (OpenML dataset ID 1597).

## Modeling Approach

The training pipeline:

1. Creates stratified train, validation and test splits.
2. Applies numerical preprocessing including imputation and scaling.
3. Compares Logistic Regression and Random Forest models with class weighting.
4. Evaluates models using Precision, Recall, F1-score, ROC-AUC and PR-AUC.
5. Tunes the decision threshold using the validation set.
6. Selects and saves the best-performing pipeline and decision threshold.

Accuracy is deliberately not treated as the primary metric because the fraudulent class is extremely rare.

## Evaluation Metrics

The project focuses on metrics that are more informative for highly imbalanced fraud detection:

* **Precision:** The proportion of transactions predicted as fraudulent that are actually fraudulent.
* **Recall:** The proportion of actual fraudulent transactions that are detected.
* **F1-score:** The harmonic mean of precision and recall.
* **PR-AUC (Average Precision):** Summarizes the precision-recall trade-off across classification thresholds.
* **ROC-AUC:** Measures the model's ability to distinguish between fraudulent and legitimate transactions across thresholds.

## Results

Results will be added after training and evaluation.

The final results will include:

* Precision
* Recall
* F1-score
* PR-AUC
* ROC-AUC
* Selected decision threshold

The actual values will be based on the model's test-set performance.

## Key ML Concepts

This project demonstrates practical understanding of:

* Highly imbalanced classification
* Precision-recall trade-offs
* Class weighting
* Decision-threshold tuning
* Stratified train/validation/test splitting
* Model comparison
* PR-AUC and ROC-AUC
* False positives and false negatives
* Data leakage prevention
* Reproducible machine learning pipelines

## Important Limitations

This is a portfolio project using an anonymized public benchmark dataset.

A production fraud-detection system would require additional considerations such as:

* Temporal validation
* Real-time or streaming infrastructure
* Feature stores
* Model and data monitoring
* Concept-drift detection
* Prediction-latency requirements
* Investigator feedback
* Business-specific cost functions
* Continuous model retraining and evaluation

These considerations are important because real-world fraud patterns change over time and the costs of false positives and false negatives can differ significantly.

## Future Improvements

Possible improvements include:

* Temporal train/validation/test splitting
* Hyperparameter tuning
* Cost-sensitive threshold optimization
* Advanced ensemble models
* Real-time fraud scoring
* Model explainability
* Monitoring for concept drift
* Integration of investigator feedback
* Production-oriented model serving

## Disclaimer

This is a portfolio/learning project using a public anonymized benchmark dataset. Its results should not be interpreted as production-level fraud-detection performance.
