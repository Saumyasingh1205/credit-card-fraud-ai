from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "creditcard.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_PATH = MODEL_DIR / "random_forest_final.joblib"
THRESHOLD_PATH = MODEL_DIR / "threshold.json"
METRICS_PATH = OUTPUT_DIR / "random_forest_metrics.json"

THRESHOLD = 0.35


def load_data():
    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print(f"Original dataset shape: {df.shape}")

    df = df.drop_duplicates()

    print(f"After removing duplicates: {df.shape}")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    return X, y


def evaluate_model(y_true, predictions):
    return {
        "accuracy": accuracy_score(
            y_true,
            predictions
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            predictions
        ).tolist()
    }


def main():

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    X, y = load_data()

    print("\nSplitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")

    print("\nTraining final Random Forest...")

    rf_final = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    rf_final.fit(
        X_train,
        y_train
    )

    print("Final Random Forest trained.")

    print(f"\nUsing selected threshold: {THRESHOLD}")

    test_probabilities = rf_final.predict_proba(
        X_test
    )[:, 1]

    test_predictions = (
        test_probabilities >= THRESHOLD
    ).astype(int)

    metrics = evaluate_model(
        y_test,
        test_predictions
    )

    print("\nFINAL TEST RESULTS")
    print("------------------")
    print(f"Threshold : {THRESHOLD}")
    print(f"Accuracy  : {metrics['accuracy']:.6f}")
    print(f"Precision : {metrics['precision']:.6f}")
    print(f"Recall    : {metrics['recall']:.6f}")
    print(f"F1 Score  : {metrics['f1']:.6f}")

    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])

    print("\nSaving model...")

    joblib.dump(
        rf_final,
        MODEL_PATH
    )

    print(f"Model saved to: {MODEL_PATH}")

    threshold_data = {
        "threshold": THRESHOLD
    }

    with open(
        THRESHOLD_PATH,
        "w"
    ) as f:
        json.dump(
            threshold_data,
            f,
            indent=4
        )

    print(f"Threshold saved to: {THRESHOLD_PATH}")

    metrics["threshold"] = THRESHOLD

    with open(
        METRICS_PATH,
        "w"
    ) as f:
        json.dump(
            metrics,
            f,
            indent=4
        )

    print(f"Metrics saved to: {METRICS_PATH}")

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()