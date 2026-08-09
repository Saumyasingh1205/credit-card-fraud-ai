from pathlib import Path
import json
import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "random_forest_final.joblib"
THRESHOLD_PATH = BASE_DIR / "models" / "threshold.json"


def load_model():
    model = joblib.load(MODEL_PATH)

    with open(THRESHOLD_PATH, "r") as f:
        threshold_data = json.load(f)

    threshold = threshold_data["threshold"]

    return model, threshold


def predict_transaction(transaction):
    model, threshold = load_model()

    if isinstance(transaction, dict):
        transaction = pd.DataFrame([transaction])

    elif isinstance(transaction, pd.Series):
        transaction = transaction.to_frame().T

    elif not isinstance(transaction, pd.DataFrame):
        raise TypeError(
            "Transaction must be a dictionary, pandas Series, "
            "or pandas DataFrame."
        )

    expected_features = [
        "Time",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
        "V7",
        "V8",
        "V9",
        "V10",
        "V11",
        "V12",
        "V13",
        "V14",
        "V15",
        "V16",
        "V17",
        "V18",
        "V19",
        "V20",
        "V21",
        "V22",
        "V23",
        "V24",
        "V25",
        "V26",
        "V27",
        "V28",
        "Amount"
    ]

    missing_features = [
        feature
        for feature in expected_features
        if feature not in transaction.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    transaction = transaction[expected_features]

    probability = model.predict_proba(
        transaction
    )[:, 1][0]

    prediction = int(
        probability >= threshold
    )

    return {
        "prediction": prediction,
        "fraud_probability": float(probability),
        "threshold": float(threshold),
        "label": (
            "Fraudulent"
            if prediction == 1
            else "Legitimate"
        )
    }


if __name__ == "__main__":

    sample_transaction = {
        "Time": 0,
        "V1": 1.0,
        "V2": 0.0,
        "V3": 1.0,
        "V4": 0.0,
        "V5": 0.0,
        "V6": 0.0,
        "V7": 0.0,
        "V8": 0.0,
        "V9": 0.0,
        "V10": 0.0,
        "V11": 0.0,
        "V12": 0.0,
        "V13": 0.0,
        "V14": 0.0,
        "V15": 0.0,
        "V16": 0.0,
        "V17": 0.0,
        "V18": 0.0,
        "V19": 0.0,
        "V20": 0.0,
        "V21": 0.0,
        "V22": 0.0,
        "V23": 0.0,
        "V24": 0.0,
        "V25": 0.0,
        "V26": 0.0,
        "V27": 0.0,
        "V28": 0.0,
        "Amount": 100.0
    }

    result = predict_transaction(
        sample_transaction
    )

    print("\nPrediction Result")
    print("-----------------")
    print(f"Prediction        : {result['label']}")
    print(
        f"Fraud Probability : "
        f"{result['fraud_probability']:.6f}"
    )
    print(
        f"Threshold         : "
        f"{result['threshold']:.2f}"
    )