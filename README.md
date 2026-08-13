# 💳 Credit Card Fraud Detection & Explainability

An end-to-end machine learning application for detecting potentially fraudulent credit card transactions and explaining individual predictions using **SHAP** and a **Groq-powered LLM**.

The project combines **machine learning, model explainability, generative AI, and Streamlit deployment** into a single interactive application.

### 🔗 Links

* 🚀 **Live Demo:** https://credit-card-fraud-ai.streamlit.app/
* 💻 **GitHub:** https://github.com/Saumyasingh1205/credit-card-fraud-ai

---

## ✨ Features

### 🔍 Fraud Detection

* Predicts whether a credit card transaction is **fraudulent or legitimate**.
* Supports **Logistic Regression** and **Random Forest** during model development.
* Uses the final **Random Forest model** for application-level predictions.
* Applies a configurable classification threshold to determine the final prediction.

### 🌲 Random Forest Prediction

The deployed application uses the trained Random Forest model to calculate the probability that a transaction is fraudulent.

The final classification is determined using the configured threshold:

```text
Fraud Probability >= Threshold → Fraudulent
Fraud Probability < Threshold  → Legitimate
```

### 📊 SHAP Explainability

SHAP is used to explain individual Random Forest predictions.

For each transaction, the application identifies the features that contributed most strongly to the prediction and shows whether each feature pushed the model toward or away from fraud.

The interface provides:

* Feature name
* Feature value
* SHAP contribution
* Direction of influence
* SHAP contribution visualization

This makes the model's decision more transparent than providing only a prediction label.

### 🤖 LLM-Powered Explanations

The project uses a **Groq-hosted LLM** through an OpenAI-compatible API to convert SHAP results into a concise, human-readable explanation.

The LLM receives information such as:

* Random Forest prediction
* Fraud probability
* Classification threshold
* Important SHAP contributions

The LLM then explains the model's decision in natural language.

> **Important:** The LLM does **not** detect fraud.
> The Random Forest model is solely responsible for the prediction.

### 🎲 Sample Transactions

The application includes predefined sample transactions for demonstration.

Users can:

* Analyze a sample fraudulent transaction
* Analyze a sample legitimate transaction
* Explore different randomly selected examples

This allows users to try the application without manually entering all transaction features.

### ✍️ Manual Transaction Input

Users can manually enter:

* `Time`
* `Amount`
* `V1`–`V28`

The transaction is then passed through the complete prediction and explainability pipeline.

### 🎨 Interactive Streamlit Interface

The application brings the complete workflow into one interface, including:

* Transaction input
* Fraud probability
* Classification threshold
* Prediction result
* SHAP feature contributions
* SHAP visualization
* AI-generated explanation

### ☁️ Cloud Deployment

The application is publicly deployed using **Streamlit Cloud**.

---

## 🧠 System Architecture

The overall prediction and explanation pipeline is:

```text
                    Transaction Input
                           │
                           ▼
                  Data Preparation
                           │
                           ▼
                  Random Forest Model
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
       Fraud Probability        Model Prediction
                │                     │
                └──────────┬──────────┘
                           ▼
                Classification Threshold
                           │
                           ▼
                 Fraud / Legitimate
                           │
                           ▼
                    SHAP Explainer
                           │
                           ▼
               Feature Contributions
                           │
                           ▼
                      Groq LLM
                           │
                           ▼
               Human-Readable Explanation
```

### Design Principle

The project intentionally separates **prediction** from **explanation**:

```text
Random Forest
      │
      │ Prediction
      ▼
    SHAP
      │
      │ Feature Contributions
      ▼
  Groq LLM
      │
      │ Natural-Language Explanation
      ▼
    User
```

This ensures that the generative AI layer explains the machine learning model's decision rather than independently making the fraud classification.

---

## 📊 Dataset

The project uses the **Credit Card Fraud Detection** dataset containing anonymized transaction features.

### Features

| Feature    | Description                              |
| ---------- | ---------------------------------------- |
| `Time`     | Time elapsed since the first transaction |
| `V1`–`V28` | Anonymized PCA-transformed features      |
| `Amount`   | Transaction amount                       |
| `Class`    | Target variable                          |

### Target Variable

```text
0 → Legitimate
1 → Fraudulent
```

The `V1`–`V28` features are anonymized PCA components. Their real-world meanings are **not assumed or inferred** by the application.

The original dataset is used for model development and evaluation.

For the deployed application, a separate JSON file containing representative sample transactions is used to provide convenient demonstration inputs.

---

## 🔬 Explainability with SHAP

The application uses **SHAP (SHapley Additive exPlanations)** to interpret individual Random Forest predictions.

For a given transaction:

* **Positive SHAP value** → pushes the prediction toward fraud
* **Negative SHAP value** → pushes the prediction away from fraud

The application identifies the most influential features and presents their contributions visually.

Example interpretation:

```text
Feature      SHAP Value       Influence
-----------------------------------------
V14          +0.31            Toward fraud
V4           +0.18            Toward fraud
Amount       -0.09            Away from fraud
V12          -0.07            Away from fraud
```

The exact contribution values depend on the transaction being analyzed.

---

## 🤖 LLM Explanation Layer

The LLM acts as an **explanation layer**, not a prediction layer.

### Input to the LLM

The explanation module receives:

```text
Random Forest Prediction
Fraud Probability
Classification Threshold
Important SHAP Contributions
```

### Output

The LLM converts these technical model outputs into a short natural-language explanation that is easier for a user to understand.

### Why this architecture?

Separating the prediction and explanation responsibilities provides a clearer architecture:

```text
Machine Learning Model
        ↓
Actual Prediction
        ↓
SHAP
        ↓
Feature-Level Evidence
        ↓
LLM
        ↓
Human-Readable Explanation
```

The LLM therefore does not replace the fraud detection model.

---

## 🖥️ Application Workflow

The Streamlit application supports three ways to analyze a transaction.

### 1. Manual Transaction

Users enter the transaction features manually:

```text
Time
Amount
V1
V2
...
V28
```

The transaction is then passed through the complete ML and explainability pipeline.

### 2. Sample Fraud Transaction

The application randomly selects a predefined fraudulent transaction from the sample dataset.

### 3. Sample Legitimate Transaction

The application randomly selects a predefined legitimate transaction from the sample dataset.

Sample transactions make it easy to explore the application without manually entering all 30 features.

---

## 📈 Prediction Output

For every analyzed transaction, the application displays:

* **Prediction label**
* **Fraud probability**
* **Classification threshold**
* **SHAP feature contributions**
* **SHAP visualization**
* **AI-generated explanation**

Example:

```text
Prediction: Legitimate
Fraud Probability: 0%
Threshold: 0.35
```

The final classification is determined by the Random Forest model and the configured threshold.

> **Note:** A model probability should not automatically be interpreted as a calibrated real-world probability of fraud. Proper probability calibration and domain-specific validation would be required for production use.

---

## 🛠️ Tech Stack

| Category             | Technologies                       |
| -------------------- | ---------------------------------- |
| **Language**         | Python                             |
| **Data Processing**  | Pandas, NumPy                      |
| **Machine Learning** | Scikit-learn                       |
| **Models**           | Logistic Regression, Random Forest |
| **Explainability**   | SHAP                               |
| **Generative AI**    | Groq API                           |
| **Application**      | Streamlit                          |
| **Visualization**    | Plotly                             |
| **Version Control**  | Git, GitHub                        |
| **Deployment**       | Streamlit Cloud                    |

---

## 📁 Project Structure

```text
credit-card-fraud-ai/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   └── creditcard.csv
│   │
│   └── sample/
│       └── sample_transactions.json
│
├── models/
│   └── random_forest/
│       └── random_forest_final.joblib
│
├── notebooks/
│   └── ...
│
├── src/
│   ├── predict.py
│   ├── explain.py
│   └── llm.py
│
├── tests/
│   └── test_llm.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Saumyasingh1205/credit-card-fraud-ai.git
cd credit-card-fraud-ai
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure the Groq API Key

The LLM explanation module requires a Groq API key.

Create the following file:

```text
.streamlit/secrets.toml
```

Add:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

> **Security:** Never commit your API key to GitHub.

---

## ▶️ Run the Application

From the project root:

```bash
python -m streamlit run app/app.py
```

The application will open in your browser.

---

## 🔐 Environment & Secrets

The project uses **Streamlit secrets** for API credentials.

Required secret:

```text
GROQ_API_KEY
```

For Streamlit Cloud deployment, configure the API key through the platform's **Secrets** settings rather than storing it inside the repository.

---

## 🧪 Testing

The project includes testing for the LLM explanation module.

Run:

```bash
python tests/test_llm.py
```

The application can also be tested manually through the Streamlit interface using:

* Sample fraud transactions
* Sample legitimate transactions
* Manual transaction inputs

---

## 🌐 Deployment

The application is deployed using **Streamlit Cloud**.

### Live Application

🚀 **https://credit-card-fraud-ai.streamlit.app/**

Changes pushed to the configured GitHub branch can trigger a new deployment.

---

## 🔮 Future Improvements

Potential improvements include:

* Improve the manual transaction input experience
* Add automatic validation for transaction feature values
* Add more representative sample transactions
* Improve model calibration and probability interpretation
* Add additional model comparison visualizations
* Expand automated test coverage
* Improve application monitoring and logging
* Add richer interactive SHAP visualizations
* Improve UI/UX based on user feedback

---

## 👥 Contributors

This project was developed collaboratively using Git and GitHub with feature-based development and Git branches.

### Aryamann Srivastava

[GitHub](https://github.com/Aryamann687)

### Saumya Singh

[GitHub](https://github.com/Saumyasingh1205)

### Areas of Contribution

The project covers collaborative work across:

* Data analysis and preprocessing
* Machine learning models
* Random Forest pipeline
* Model evaluation
* SHAP explainability
* LLM-based explanations
* Streamlit application
* Deployment

---

## ⚠️ Disclaimer

This project is intended for **educational and demonstration purposes**.

It should **not** be used as a production financial fraud detection system without additional:

* Model validation
* Probability calibration
* Security controls
* Monitoring
* Data validation
* Domain-specific evaluation
* Regulatory and compliance considerations

The anonymized `V1`–`V28` features do not have publicly interpretable real-world meanings.

---

## 📄 License

This project is intended for **educational and portfolio purposes**.

