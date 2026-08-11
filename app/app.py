import sys
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# PROJECT PATH
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.predict import predict_transaction
from src.explain import explain_transaction
from src.llm import generate_explanation


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# COLORS
# ============================================================

COLORS = {
    "bg": "#070B14",
    "panel": "#0D1422",
    "panel2": "#111A2B",
    "border": "#263247",
    "blue": "#38BDF8",
    "blue_dark": "#2563EB",
    "text": "#F8FAFC",
    "muted": "#94A3B8",
    "fraud": "#F87171",
    "legit": "#34D399",
    "warning": "#FBBF24",
}


V_FEATURES = [f"V{i}" for i in range(1, 29)]

FEATURES = [
    "Time",
    *V_FEATURES,
    "Amount",
]


# ============================================================
# GLOBAL STYLING
# ============================================================

st.markdown(
    f"""
    <style>

    .stApp {{
        background:
            radial-gradient(
                circle at 15% 0%,
                rgba(37, 99, 235, 0.12),
                transparent 35%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(56, 189, 248, 0.06),
                transparent 30%
            ),
            {COLORS["bg"]};
    }}

    .block-container {{
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }}

    header[data-testid="stHeader"] {{
        background: transparent;
    }}

    #MainMenu {{
        visibility: hidden;
    }}

    footer {{
        visibility: hidden;
    }}

    * {{
        font-family:
            Inter,
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            sans-serif;
    }}

    h1, h2, h3, h4 {{
        color: {COLORS["text"]} !important;
        letter-spacing: -0.025em;
    }}

    h1 {{
        font-size: 3.2rem !important;
        font-weight: 750 !important;
        line-height: 1.1 !important;
    }}

    h2 {{
        font-size: 2rem !important;
        font-weight: 700 !important;
    }}

    h3 {{
        font-size: 1.35rem !important;
        font-weight: 700 !important;
    }}

    p {{
        color: {COLORS["muted"]};
    }}

    hr {{
        border: none;
        border-top: 1px solid {COLORS["border"]};
        margin: 2rem 0;
    }}

    /* ========================================================
       RADIO BUTTONS
       ======================================================== */

    div[role="radiogroup"] {{
        gap: 0.7rem;
    }}

    div[role="radiogroup"] label {{
        background: transparent;
        border: 1px solid transparent;
        border-radius: 8px;
        padding: 0.35rem 0.6rem;
    }}

    /* ========================================================
       NUMBER INPUTS
       ======================================================== */

    div[data-testid="stNumberInput"] input {{
        background: {COLORS["panel"]} !important;
        color: {COLORS["text"]} !important;
        border: 1px solid {COLORS["border"]} !important;
        border-radius: 9px !important;
        font-weight: 600 !important;
    }}

    div[data-testid="stNumberInput"] button {{
        background: {COLORS["panel"]} !important;
        border-color: {COLORS["border"]} !important;
        color: {COLORS["text"]} !important;
    }}

   /* ========================================================
   ANALYZE TRANSACTION BUTTON
   ======================================================== */

div[data-testid="stButton"] > button {{
    border: none !important;
    border-radius: 10px !important;
    min-height: 3.2rem !important;

    background: linear-gradient(
        135deg,
        #7C3AED 0%,
        #4F46E5 100%
    ) !important;

    color: white !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    letter-spacing: 0.02em;

    box-shadow: 0 8px 25px rgba(124, 58, 237, 0.25);

    transition:
        transform 0.2s ease,
        box-shadow 0.2s ease,
        filter 0.2s ease;
}}
div[data-testid="stButton"] > button:hover {{
    filter: brightness(1.08);
    transform: translateY(-2px);
    box-shadow: 0 12px 30px rgba(79, 70, 229, 0.35);
}}

div[data-testid="stButton"] > button:active {{
    transform: translateY(0);
}}

    /* ========================================================
       EXPANDER
       ======================================================== */

    div[data-testid="stExpander"] {{
        background: {COLORS["panel"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 10px;
    }}

    /* ========================================================
       METRICS
       ======================================================== */

    div[data-testid="stMetric"] {{
        background: {COLORS["panel"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 12px;
        padding: 1.2rem;
        min-height: 120px;
    }}

    div[data-testid="stMetricLabel"] {{
        color: {COLORS["muted"]} !important;
        font-size: 0.72rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}

    div[data-testid="stMetricValue"] {{
        color: {COLORS["text"]} !important;
        font-weight: 700 !important;
    }}

    /* ========================================================
       DATAFRAME
       ======================================================== */

    div[data-testid="stDataFrame"] {{
        border: 1px solid {COLORS["border"]};
        border-radius: 10px;
        overflow: hidden;
    }}

    /* ========================================================
       ALERTS
       ======================================================== */

    div[data-testid="stAlert"] {{
        border-radius: 10px;
    }}

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "result" not in st.session_state:
    st.session_state.result = None

if "error" not in st.session_state:
    st.session_state.error = None


# ============================================================
# DATASET
# ============================================================

DATA_PATH = (
    ROOT_DIR
    / "data"
    / "raw"
    / "creditcard.csv"
)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def get_sample_transactions():

    df = load_dataset()

    fraud_rows = df[df["Class"] == 1]

    legitimate_rows = df[df["Class"] == 0]

    if fraud_rows.empty:
        raise ValueError(
            "No fraudulent transactions were found in the dataset."
        )

    if legitimate_rows.empty:
        raise ValueError(
            "No legitimate transactions were found in the dataset."
        )

    fraud = (
        fraud_rows
        .iloc[0]
        .drop(labels=["Class"])
        .to_dict()
    )

    legitimate = (
        legitimate_rows
        .iloc[0]
        .drop(labels=["Class"])
        .to_dict()
    )

    return fraud, legitimate


# ============================================================
# PIPELINE
# ============================================================

def run_pipeline(transaction):

    prediction_result = predict_transaction(
        transaction
    )

    explanation_result = explain_transaction(
        transaction,
        top_n=10
    )

    explanation_text = generate_explanation(
        prediction_result,
        explanation_result
    )

    return (
        prediction_result,
        explanation_result,
        explanation_text
    )


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns(
    [3, 1],
    vertical_alignment="center"
)


with header_left:

    st.markdown(
        f"""
        <div style="
            font-size:1.1rem;
            font-weight:800;
            letter-spacing:0.08em;
            color:{COLORS["text"]};
        ">
            FRAUDGUARD AI
        </div>

        <div style="
            color:{COLORS["muted"]};
            font-size:0.75rem;
            letter-spacing:0.08em;
            margin-top:3px;
        ">
            TRANSACTION INTELLIGENCE PLATFORM
        </div>
        """,
        unsafe_allow_html=True
    )


with header_right:

    st.markdown(
        f"""
        <div style="
            text-align:right;
            padding-top:5px;
            font-size:0.78rem;
            color:{COLORS["muted"]};
        ">
            <span style="
                color:{COLORS["legit"]};
                font-size:0.9rem;
            ">●</span>
            SYSTEM OPERATIONAL
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown("<hr>", unsafe_allow_html=True)


# ============================================================
# HERO
# ============================================================

hero_left, hero_right = st.columns(
    [2.2, 1],
    gap="large"
)


with hero_left:

    st.caption(
        "AI-POWERED TRANSACTION SECURITY"
    )

    st.title(
        "Credit Card Fraud Detection"
    )

    st.markdown(
        """
        Intelligent transaction risk analysis powered by
        **Random Forest**, **SHAP explainability**, and
        **AI-assisted reasoning**.
        """
    )

    st.write("")

    stat1, stat2, stat3 = st.columns(3)

    with stat1:
        st.caption("MODEL")
        st.markdown("**Random Forest**")

    with stat2:
        st.caption("EXPLAINABILITY")
        st.markdown("**SHAP**")

    with stat3:
        st.caption("AI LAYER")
        st.markdown("**Groq LLM**")


with hero_right:

    with st.container(border=True):

        st.caption("SYSTEM")

        st.markdown(
            "**:green[● Operational]**"
        )

        st.write("")

        st.caption("CLASSIFIER")

        st.markdown(
            "**Random Forest**"
        )

        st.write("")

        st.caption("EXPLANATION")

        st.markdown(
            "**SHAP + Groq**"
        )


# ============================================================
# DETECTION PIPELINE
# ============================================================

st.write("")

st.markdown(
    "### Detection Pipeline"
)

st.caption(
    "Transaction intelligence workflow"
)


pipeline_cols = st.columns(
    5,
    gap="small"
)


pipeline_items = [
    ("01", "Transaction", "Input"),
    ("02", "Random Forest", "Classification"),
    ("03", "Risk Score", "Probability"),
    ("04", "SHAP", "Explainability"),
    ("05", "AI Insight", "Interpretation"),
]


for col, (number, title, subtitle) in zip(
    pipeline_cols,
    pipeline_items
):

    with col:

        with st.container(border=True):

            st.caption(number)

            st.markdown(
                f"**{title}**"
            )

            st.caption(
                subtitle
            )


# ============================================================
# TRANSACTION ANALYSIS
# ============================================================

st.write("")

st.markdown(
    "### Transaction Analysis"
)

st.caption(
    "Configure a transaction and run the complete risk analysis."
)

st.write("")


source = st.radio(
    "Transaction source",
    [
        "Manual Transaction",
        "Sample Fraud Transaction",
        "Sample Legitimate Transaction",
    ],
    horizontal=True,
)


# Load samples only after source selection
fraud_sample, legit_sample = (
    get_sample_transactions()
)


# ============================================================
# SOURCE DATA
# ============================================================

if source == "Sample Fraud Transaction":

    source_data = fraud_sample

    st.info(
        "Using a known fraudulent transaction from the dataset."
    )

elif source == "Sample Legitimate Transaction":

    source_data = legit_sample

    st.info(
        "Using a known legitimate transaction from the dataset."
    )

else:

    source_data = {
        feature: 0.0
        for feature in FEATURES
    }

    st.info(
        "Enter transaction values manually. "
        "V1–V28 are anonymized PCA-based model features."
    )


# ============================================================
# TRANSACTION DETAILS
# ============================================================

st.markdown(
    "#### Transaction Details"
)


time_col, amount_col = st.columns(
    2,
    gap="large"
)


with time_col:

    transaction_time = st.number_input(
        "Transaction Time",
        value=float(
            source_data.get(
                "Time",
                0.0
            )
        ),
        step=1.0,
        format="%.2f",
        key=f"time_{source}",
    )


with amount_col:

    transaction_amount = st.number_input(
        "Transaction Amount",
        value=float(
            source_data.get(
                "Amount",
                0.0
            )
        ),
        min_value=0.0,
        step=1.0,
        format="%.2f",
        key=f"amount_{source}",
    )


# ============================================================
# ADVANCED FEATURES
# ============================================================

with st.expander(
    "Advanced Model Features — V1 to V28"
):

    st.caption(
        "These features are anonymized PCA components "
        "from the credit-card fraud dataset."
    )

    feature_values = {}

    feature_columns = st.columns(
        4,
        gap="small"
    )

    for index, feature in enumerate(
        V_FEATURES
    ):

        with feature_columns[index % 4]:

            feature_values[feature] = st.number_input(
                feature,
                value=float(
                    source_data.get(
                        feature,
                        0.0
                    )
                ),
                format="%.6f",
                key=f"{feature}_{source}",
            )


# ============================================================
# BUILD TRANSACTION
# ============================================================

transaction = {
    "Time": transaction_time,
    **feature_values,
    "Amount": transaction_amount,
}


st.write("")


analyze = st.button(
    "🔍  Analyze Transaction",
    use_container_width=True,
)


# ============================================================
# RUN MODEL
# ============================================================

if analyze:

    try:

        with st.spinner(
            "Running Random Forest → SHAP → AI analysis..."
        ):

            (
                prediction_result,
                explanation_result,
                explanation_text,
            ) = run_pipeline(transaction)

        st.session_state.result = {
            "transaction": transaction,
            "prediction": prediction_result,
            "explanation": explanation_result,
            "llm": explanation_text,
        }

        st.session_state.error = None

    except Exception:

        st.session_state.result = None

        st.session_state.error = (
            traceback.format_exc()
        )


# ============================================================
# ERROR DISPLAY
# ============================================================

if st.session_state.error:

    st.markdown(
        "<hr>",
        unsafe_allow_html=True
    )

    st.error(
        "Analysis failed. Please verify the transaction input."
    )

    with st.expander(
        "Technical details"
    ):

        st.code(
            st.session_state.error
        )


# ============================================================
# RESULTS
# ============================================================

if st.session_state.result:

    result = st.session_state.result

    prediction_result = result[
        "prediction"
    ]

    explanation_result = result[
        "explanation"
    ]

    explanation_text = result[
        "llm"
    ]


    # ========================================================
    # NORMALIZE PREDICTION
    # ========================================================

    label = prediction_result.get(
        "label",
        "Legitimate"
    )

    probability = float(
        prediction_result.get(
            "fraud_probability",
            prediction_result.get(
                "probability",
                0.0
            )
        )
    )

    threshold = float(
        prediction_result.get(
            "threshold",
            0.35
        )
    )

    is_fraud = (
        str(label).lower()
        in {
            "fraud",
            "fraudulent",
            "1",
            "true"
        }
    )


    # ========================================================
    # ANALYSIS RESULT
    # ========================================================

    st.markdown(
        "<hr>",
        unsafe_allow_html=True
    )

    st.markdown(
        "### Analysis Result"
    )


    # --------------------------------------------------------
    # STATUS CARD
    # --------------------------------------------------------

    with st.container(border=True):

        if is_fraud:

            st.error(
                "🚨 FRAUD RISK DETECTED"
            )

            st.caption(
                "The Random Forest model classified "
                "this transaction as potentially fraudulent."
            )

        else:

            st.success(
                "✓ TRANSACTION CLEARED"
            )

            st.caption(
                "The Random Forest model classified "
                "this transaction as legitimate."
            )


    st.write("")


    # ========================================================
    # KEY METRICS
    # ========================================================

    metric1, metric2, metric3 = st.columns(
        3,
        gap="medium"
    )


    with metric1:

        st.metric(
            "Classification",
            "FRAUD"
            if is_fraud
            else "LEGITIMATE"
        )


    with metric2:

        st.metric(
            "Fraud Probability",
            f"{probability:.2%}"
        )


    with metric3:

        st.metric(
            "Decision Threshold",
            f"{threshold:.2f}"
        )


    st.write("")


    # ========================================================
    # FULL-WIDTH FRAUD RISK SCORE
    # ========================================================

    st.markdown(
        "#### Fraud Risk Score"
    )

    st.caption(
        "The probability of fraud compared with the model decision threshold."
    )


    fig = go.Figure()


    bar_color = (
        COLORS["fraud"]
        if is_fraud
        else COLORS["legit"]
    )


    fig.add_trace(
        go.Bar(
            x=[
                probability * 100
            ],
            y=["Risk"],
            orientation="h",
            marker=dict(
                color=bar_color,
                line=dict(
                    width=0
                ),
            ),
            hovertemplate=(
                "Fraud probability: "
                "%{x:.2f}%"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )


    # Threshold marker
    fig.add_shape(
        type="line",
        x0=threshold * 100,
        x1=threshold * 100,
        y0=-0.45,
        y1=0.45,
        line=dict(
            color=COLORS["warning"],
            width=3,
            dash="dash",
        ),
    )


    fig.add_annotation(
        x=threshold * 100,
        y=0.5,
        text=f"Threshold {threshold:.0%}",
        showarrow=False,
        font=dict(
            color=COLORS["warning"],
            size=12,
        ),
        yshift=10,
    )


    fig.update_layout(
        height=180,
        margin=dict(
            l=10,
            r=10,
            t=25,
            b=35,
        ),
        paper_bgcolor=COLORS["panel"],
        plot_bgcolor=COLORS["panel"],
        font=dict(
            color=COLORS["muted"]
        ),
    )


    fig.update_xaxes(
        range=[0, 100],
        ticksuffix="%",
        showgrid=True,
        gridcolor=COLORS["border"],
        zeroline=False,
        color=COLORS["muted"],
    )


    fig.update_yaxes(
        visible=False
    )


    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )


    st.caption(
        f"Decision threshold: {threshold:.0%}"
    )


    # ========================================================
    # MODEL EXPLAINABILITY
    # ========================================================

    st.write("")

    st.markdown(
        "### Model Explainability"
    )

    st.caption(
        "SHAP shows which model features contributed most strongly "
        "to the prediction."
    )


    top_features = (
        explanation_result.get(
            "top_features",
            []
        )
        if isinstance(
            explanation_result,
            dict
        )
        else []
    )


    shap_df = pd.DataFrame(
        top_features
    )


    if not shap_df.empty and "shap_value" in shap_df.columns:

        shap_df["absolute_shap"] = (
            shap_df["shap_value"].abs()
        )


        shap_df = (
            shap_df
            .sort_values(
                "absolute_shap",
                ascending=False
            )
            .head(8)
        )


        left, right = st.columns(
            [1.5, 1],
            gap="large"
        )


        # ----------------------------------------------------
        # SHAP CHART
        # ----------------------------------------------------

        with left:

            st.markdown(
                "**Feature Contribution**"
            )


            chart_df = (
                shap_df
                .sort_values(
                    "shap_value"
                )
            )


            chart_colors = [
                COLORS["fraud"]
                if value > 0
                else COLORS["legit"]
                for value in chart_df[
                    "shap_value"
                ]
            ]


            fig2 = go.Figure()


            fig2.add_trace(
                go.Bar(
                    x=chart_df[
                        "shap_value"
                    ],
                    y=chart_df[
                        "feature"
                    ],
                    orientation="h",
                    marker=dict(
                        color=chart_colors
                    ),
                    hovertemplate=(
                        "%{y}: %{x:.5f}"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                )
            )


            fig2.update_layout(
                height=380,
                margin=dict(
                    l=10,
                    r=10,
                    t=10,
                    b=10,
                ),
                paper_bgcolor=COLORS["panel"],
                plot_bgcolor=COLORS["panel"],
                font=dict(
                    color=COLORS["muted"]
                ),
            )


            fig2.update_xaxes(
                title="SHAP contribution",
                showgrid=True,
                gridcolor=COLORS["border"],
                zeroline=True,
                zerolinecolor=COLORS["muted"],
            )


            fig2.update_yaxes(
                showgrid=False
            )


            st.plotly_chart(
                fig2,
                use_container_width=True,
                config={
                    "displayModeBar": False,
                    "responsive": True,
                },
            )


        # ----------------------------------------------------
        # FEATURE TABLE
        # ----------------------------------------------------

        with right:

            st.markdown(
                "**Top Contributing Features**"
            )


            display_df = shap_df[
                [
                    "feature",
                    "value",
                    "shap_value",
                ]
            ].copy()


            display_df.columns = [
                "Feature",
                "Value",
                "SHAP Impact",
            ]


            display_df["Value"] = (
                display_df["Value"]
                .round(4)
            )


            display_df["SHAP Impact"] = (
                display_df["SHAP Impact"]
                .round(6)
            )


            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )


        st.caption(
            "Positive SHAP values push the prediction toward fraud. "
            "Negative values push it away from fraud."
        )


        st.caption(
            "V1–V28 are anonymized PCA components and do not represent "
            "specific real-world transaction attributes."
        )


    else:

        st.info(
            "No SHAP contribution data available."
        )


    # ========================================================
    # AI INTERPRETATION
    # ========================================================

    st.write("")

    st.markdown(
        "### AI Interpretation"
    )

    st.caption(
        "HUMAN-READABLE EXPLANATION · POWERED BY GROQ"
    )


    # IMPORTANT:
    # The explanation is now INSIDE the container.

    with st.container(border=True):

        st.caption(
            "AI INSIGHT"
        )

        st.write(
            explanation_text
        )

        st.caption(
            "Based on the Random Forest prediction "
            "and SHAP contributions."
        )


    st.caption(
        "The AI layer explains the existing Random Forest "
        "decision using SHAP contributions. It does not "
        "independently classify the transaction."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    "<hr>",
    unsafe_allow_html=True
)


footer_left, footer_right = st.columns(
    [1, 2]
)


with footer_left:

    st.caption(
        "FRAUDGUARD AI"
    )

    st.caption(
        "AI-assisted transaction risk analysis"
    )


with footer_right:

    st.markdown(
        f"""
        <div style="
            text-align:right;
            color:{COLORS["muted"]};
            font-size:0.72rem;
            line-height:1.5;
        ">
            FraudGuard AI is a decision-support system.
            Predictions should not be treated as definitive
            proof of fraudulent activity.
        </div>
        """,
        unsafe_allow_html=True
    )