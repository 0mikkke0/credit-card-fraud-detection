from pathlib import Path
from sklearn.datasets import fetch_openml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "creditcard.csv"

print("Downloading credit-card fraud dataset from OpenML...")
dataset = fetch_openml(data_id=1597, as_frame=True, parser="auto")
df = dataset.frame.copy()

if "Class" not in df.columns and dataset.target is not None:
    df["Class"] = dataset.target

df.to_csv(OUT, index=False)
print(f"Saved {len(df):,} rows to {OUT}")
print(df["Class"].value_counts())
