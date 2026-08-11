from pathlib import Path
import pandas as pd

from src.predict import predict_transaction


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "creditcard.csv"

df = pd.read_csv(DATA_PATH)


def test_transaction(transaction, actual_class, name):

    result = predict_transaction(
        transaction
    )

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Actual Class      : {actual_class}")
    print(f"Prediction        : {result['label']}")
    print(
        f"Fraud Probability : "
        f"{result['fraud_probability']:.6f}"
    )
    print(
        f"Threshold         : "
        f"{result['threshold']:.2f}"
    )


fraud_row = df[
    df["Class"] == 1
].iloc[0]

fraud_transaction = fraud_row.drop(
    labels=["Class"]
)

test_transaction(
    fraud_transaction,
    fraud_row["Class"],
    "Fraud Transaction"
)


legitimate_row = df[
    df["Class"] == 0
].iloc[0]

legitimate_transaction = legitimate_row.drop(
    labels=["Class"]
)

test_transaction(
    legitimate_transaction,
    legitimate_row["Class"],
    "Legitimate Transaction"
)