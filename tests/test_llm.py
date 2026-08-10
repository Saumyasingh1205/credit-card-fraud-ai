import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


from src.predict import predict_transaction
from src.explain import explain_transaction
from src.llm import generate_explanation


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


prediction_result = predict_transaction(
    sample_transaction
)

explanation_result = explain_transaction(
    sample_transaction,
    top_n=5
)

print("\nPrediction:")
print(prediction_result)

print("\nSHAP:")
print(explanation_result["top_features"])

print("\nLLM Explanation:")
print(
    generate_explanation(
        prediction_result,
        explanation_result
    )
)