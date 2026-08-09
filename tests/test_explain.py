from pathlib import Path
import pandas as pd

from src.explain import explain_transaction


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "creditcard.csv"

df = pd.read_csv(DATA_PATH)


fraud_row = df[
    df["Class"] == 1
].iloc[0]

fraud_transaction = fraud_row.drop(
    labels=["Class"]
)

result = explain_transaction(
    fraud_transaction,
    top_n=10
)

print("\nFraud Transaction Explanation")
print("-----------------------------")

print(
    f"Actual Class      : "
    f"{fraud_row['Class']}"
)

print(
    f"Prediction        : "
    f"{result['label']}"
)

print(
    f"Fraud Probability : "
    f"{result['fraud_probability']:.6f}"
)

print(
    f"Threshold         : "
    f"{result['threshold']:.2f}"
)

print("\nTop contributing features:")

for item in result["top_features"]:
    direction = (
        "increased fraud probability"
        if item["shap_value"] > 0
        else "decreased fraud probability"
    )

    print(
        f"{item['feature']:>7} | "
        f"value={item['value']:<12} | "
        f"SHAP={item['shap_value']:.6f} | "
        f"{direction}"
    )