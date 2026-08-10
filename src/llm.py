import streamlit as st
from openai import OpenAI


def generate_explanation(prediction_result, explanation_result):
    client = OpenAI(
        api_key=st.secrets["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1"
    )

    prediction = prediction_result["label"]
    probability = prediction_result["fraud_probability"]
    threshold = prediction_result["threshold"]

    top_features = explanation_result["top_features"]

    feature_text = "\n".join(
        [
            (
                f"- {item['feature']}: "
                f"value={item['value']}, "
                f"SHAP={item['shap_value']:.6f}"
            )
            for item in top_features
        ]
    )

    instructions = """
You are an explanation assistant for a credit card fraud
detection system.

The fraud prediction has ALREADY been made by a Random
Forest machine learning model.

Your job is ONLY to explain the model's decision in
simple, clear language.

Rules:

1. Never make or change the fraud prediction.
2. Never claim that you independently detected fraud.
3. Use the supplied prediction and fraud probability.
4. Explain how the supplied SHAP values contributed to
   the model's prediction.
5. A positive SHAP value means the feature pushed the
   prediction toward fraud.
6. A negative SHAP value means the feature pushed the
   prediction away from fraud.
7. V1-V28 are anonymized PCA components. Do NOT invent
   real-world meanings for these features.
8. Do not claim that a feature represents a merchant,
   location, amount, device, or any other real-world
   attribute unless that information is explicitly given.
9. Keep the explanation concise and understandable to
   a non-technical user.
10. Return 2-3 short paragraphs.
11. Do not use tables.
12. Do not use headings.
13. Do not repeat every SHAP value.
14. Mention the 2-3 strongest contributing features.
"""

    input_text = f"""
Random Forest prediction: {prediction}
Fraud probability: {probability:.2%}
Classification threshold: {threshold:.2f}

Top SHAP feature contributions:
{feature_text}

Explain why the Random Forest produced this prediction.
Mention the strongest factors and whether they increased
or decreased the model's fraud probability.
"""

    response = client.responses.create(
        model="openai/gpt-oss-120b",
        instructions=instructions,
        input=input_text
    )

    return response.output_text


if __name__ == "__main__":
    print("LLM module loaded successfully.")