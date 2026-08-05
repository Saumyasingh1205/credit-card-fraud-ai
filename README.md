# 💳 AI-Powered Credit Card Fraud Detection

An end-to-end Machine Learning platform for detecting fraudulent credit card transactions using Explainable AI (SHAP) and Large Language Models (LLMs).

---

## 📌 Overview

Credit card fraud is one of the most challenging real-world machine learning problems due to its highly imbalanced nature and continuously evolving fraud patterns.

This project aims to build a production-inspired fraud detection system that combines Machine Learning, Explainable AI, and Generative AI into a single application.

Unlike a traditional classifier, this platform not only predicts whether a transaction is fraudulent but also explains **why** it was flagged using SHAP and generates human-readable explanations using an LLM.

---

# ✨ Features

### Machine Learning

- Exploratory Data Analysis (EDA)
- Data preprocessing pipeline
- Feature scaling
- Class imbalance handling (SMOTE, Class Weights, Undersampling)
- Logistic Regression baseline
- Random Forest
- XGBoost
- Hyperparameter tuning
- Cross-validation
- Model comparison

### Explainable AI

- SHAP feature importance
- Local explanations
- Global feature importance
- Prediction confidence visualization

### AI Layer

- LLM-powered fraud explanations
- Natural language summaries
- AI-assisted fraud analysis
- Interactive chatbot

### Web Application

- Upload transaction datasets
- Manual transaction prediction
- Fraud probability prediction
- SHAP explanation dashboard
- Download prediction reports

---

# 🏗️ System Architecture

```
                    User
                      │
                      ▼
              Streamlit Dashboard
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
 Machine Learning Model      LLM Explanation
          │                       │
          ▼                       ▼
    SHAP Explainability     Natural Language
          │                       │
          └───────────┬───────────┘
                      ▼
              Fraud Analysis Report
```

---

# 📂 Project Structure

```
credit-card-fraud-ai/

│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│
├── models/
│
├── notebooks/
│   └── 01_EDA.ipynb
│
├── outputs/
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluation.py
│   └── explain.py
│
├── tests/
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

# 📊 Dataset

This project uses the **Credit Card Fraud Detection** dataset published by the Machine Learning Group (ULB) and made available on Kaggle.

### Dataset Summary

- **Total Transactions:** 284,807
- **Fraudulent Transactions:** 492
- **Legitimate Transactions:** 284,315
- **Fraud Rate:** 0.172%
- **Features:** 31
- **Target Variable:** `Class`
- **File Format:** CSV

### Feature Description

| Feature | Description |
|----------|-------------|
| Time | Seconds elapsed between this transaction and the first transaction |
| V1 - V28 | PCA-transformed anonymized features |
| Amount | Transaction amount |
| Class | Target variable (0 = Genuine, 1 = Fraudulent) |

> Due to confidentiality reasons, the original features were transformed using PCA. Only **Time** and **Amount** remain in their original form. The dataset is highly imbalanced, making metrics such as Precision-Recall AUC more informative than accuracy alone.

Dataset Source:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

# 🛠️ Technology Stack

## Programming

- Python

## Data Processing

- NumPy
- Pandas

## Visualization

- Matplotlib
- Seaborn

## Machine Learning

- Scikit-Learn
- XGBoost
- Imbalanced-Learn

## Explainability

- SHAP

## AI Integration

- OpenAI API

## Deployment

- Streamlit

## Version Control

- Git
- GitHub

---

# 🔄 Machine Learning Pipeline

```
Raw Dataset
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Data Preprocessing
      │
      ▼
Feature Scaling
      │
      ▼
Handle Class Imbalance
      │
      ▼
Model Training
      │
      ▼
Model Evaluation
      │
      ▼
SHAP Explainability
      │
      ▼
LLM Explanation
      │
      ▼
Streamlit Deployment
```

---

# 📈 Model Evaluation

Models will be evaluated using:

- Precision
- Recall
- F1 Score
- ROC-AUC
- Precision-Recall AUC
- Confusion Matrix
- Cross Validation

The best-performing model will be selected based on fraud detection performance rather than overall accuracy.

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/Saumyasingh1205/credit-card-fraud-ai.git

cd credit-card-fraud-ai
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment:

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

```bash
streamlit run app/streamlit_app.py
```

---

# 🗺️ Development Roadmap

- [x] Repository setup
- [x] Professional project structure
- [x] Git workflow
- [ ] Dataset exploration
- [ ] Exploratory Data Analysis
- [ ] Data preprocessing
- [ ] Feature engineering
- [ ] Handling class imbalance
- [ ] Baseline model
- [ ] Random Forest
- [ ] XGBoost
- [ ] Hyperparameter tuning
- [ ] SHAP integration
- [ ] Streamlit dashboard
- [ ] LLM integration
- [ ] AI chatbot
- [ ] Deployment
- [ ] Documentation

---

# 🌳 Git Workflow

```
main
│
└── develop
      │
      ├── feature/01-eda
      ├── feature/02-preprocessing
      ├── feature/03-model-training
      ├── feature/04-model-evaluation
      ├── feature/05-xgboost
      ├── feature/06-hyperparameter-tuning
      ├── feature/07-shap
      ├── feature/08-streamlit-ui
      ├── feature/09-llm-integration
      ├── feature/10-chatbot
      └── feature/11-deployment
```

---

# 📚 Learning Outcomes

This project demonstrates practical experience in:

- End-to-End Machine Learning
- Feature Engineering
- Imbalanced Learning
- Model Evaluation
- Explainable AI
- Large Language Models
- Streamlit Deployment
- Git Collaboration
- Software Engineering Best Practices

---

# 👥 Contributors

| Name | Role |
|------|------|
| **Saumya Singh** | Machine Learning Engineer |
| **Aryaman Srivastava** | Machine Learning Engineer |

---

# 📄 License

This project is licensed under the MIT License.

---

# ⭐ Status

🚧 **Currently under active development.**

Future updates will include model performance benchmarks, SHAP visualizations, Streamlit screenshots, deployment links, and a demonstration video.