import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.predict import predict_transaction
from src.explain import explain_transaction
from src.llm import generate_explanation


DATA_PATH = ROOT_DIR / "data" / "raw" / "creditcard.csv"


st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)


st.title("💳 Credit Card Fraud Detection System")

st.write(
    "AI-powered credit card fraud detection using "
    "Random Forest and SHAP."
)

st.divider()


st.subheader("Choose Transaction")

transaction_type = st.radio(
    "Select input method:",
    [
        "Manual Transaction",
        "Sample Fraud Transaction",
        "Sample Legitimate Transaction"
    ],
    horizontal=True
)


transaction = None


if transaction_type == "Manual Transaction":

    st.subheader("Transaction Details")

    col1, col2 = st.columns(2)

    with col1:
        time = st.number_input(
            "Time",
            value=0.0
        )

    with col2:
        amount = st.number_input(
            "Amount",
            value=100.0,
            min_value=0.0
        )

    st.subheader("Transaction Features")

    features = {}

    cols = st.columns(4)

    for i in range(1, 29):

        with cols[(i - 1) % 4]:

            features[f"V{i}"] = st.number_input(
                f"V{i}",
                value=0.0,
                format="%.6f"
            )

    transaction = {
        "Time": time,
        **features,
        "Amount": amount
    }


elif transaction_type == "Sample Fraud Transaction":

    st.info(
        "Using the first known fraudulent transaction "
        "from the dataset."
    )

    df = pd.read_csv(DATA_PATH)

    fraud_row = df[
        df["Class"] == 1
    ].iloc[0]

    transaction = fraud_row.drop(
        labels=["Class"]
    )

    st.write(
        f"Actual class in dataset: **Fraudulent (1)**"
    )


elif transaction_type == "Sample Legitimate Transaction":

    st.info(
        "Using the first legitimate transaction "
        "from the dataset."
    )

    df = pd.read_csv(DATA_PATH)

    legitimate_row = df[
        df["Class"] == 0
    ].iloc[0]

    transaction = legitimate_row.drop(
        labels=["Class"]
    )

    st.write(
        f"Actual class in dataset: **Legitimate (0)**"
    )


st.divider()


if st.button(
    "🔍 Analyze Transaction",
    use_container_width=True
):

    try:

        prediction_result = predict_transaction(
            transaction
        )

        explanation_result = explain_transaction(
            transaction,
            top_n=10
        )
        llm_explanation = generate_explanation(
            prediction_result,
            explanation_result
        )


        probability = prediction_result[
            "fraud_probability"
        ]

        threshold = prediction_result[
            "threshold"
        ]

        label = prediction_result[
            "label"
        ]


        st.subheader("Prediction Result")

        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Prediction",
                label
            )


        with col2:

            st.metric(
                "Fraud Probability",
                f"{probability:.2%}"
            )


        with col3:

            st.metric(
                "Classification Threshold",
                f"{threshold:.2f}"
            )


        if label == "Fraudulent":

            st.error(
                "🚨 This transaction is classified "
                "as FRAUDULENT."
            )

        else:

            st.success(
                "✅ This transaction is classified "
                "as LEGITIMATE."
            )


        st.progress(
            min(probability, 1.0),
            text=f"Fraud Probability: {probability:.2%}"
        )


        st.divider()


        st.subheader(
            "🔎 Why did the model make this prediction?"
        )

        st.write(
            "SHAP identifies which features contributed "
            "most strongly to the prediction."
        )


        top_features = explanation_result[
            "top_features"
        ]


        shap_data = []

        for item in top_features:

            feature = item["feature"]
            value = item["value"]
            shap_value = item["shap_value"]

            if shap_value > 0:

                effect = "Increased fraud probability"

            else:

                effect = "Decreased fraud probability"


            shap_data.append(
                {
                    "Feature": feature,
                    "Value": value,
                    "SHAP Value": shap_value,
                    "Effect": effect
                }
            )


        shap_df = pd.DataFrame(
            shap_data
        )


        st.dataframe(
            shap_df,
            use_container_width=True,
            hide_index=True
        )


        st.subheader(
            "SHAP Feature Contributions"
        )


        chart_data = shap_df[
            ["Feature", "SHAP Value"]
        ].set_index("Feature")


        st.bar_chart(
            chart_data
        )
        st.divider()
        st.subheader("🤖 AI Explanation")
        st.write(llm_explanation)




        st.divider()


        st.subheader(
            "📋 Transaction Summary"
        )


        summary_col1, summary_col2 = st.columns(2)


        with summary_col1:

            st.write(
                f"**Prediction:** {label}"
            )

            st.write(
                f"**Fraud Probability:** "
                f"{probability:.6f}"
            )


        with summary_col2:

            st.write(
                f"**Threshold:** {threshold:.2f}"
            )

            st.write(
                f"**Features analyzed:** 30"
            )


    except Exception as e:

        st.error(
            f"Error while analyzing transaction: {e}"
        )