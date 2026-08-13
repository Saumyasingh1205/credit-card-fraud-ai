# 💳 Credit Card Fraud Detection & Explainability

An end-to-end machine learning system for detecting potentially fraudulent credit card transactions and explaining individual predictions using **SHAP** and a **Groq-powered LLM**.

The project combines machine learning, model explainability, generative AI, and Streamlit deployment into a single interactive application.

🔗 **Live Demo:** https://credit-card-fraud-ai.streamlit.app/  
🔗 **GitHub:** https://github.com/Saumyasingh1205/credit-card-fraud-ai

---

## 🚀 Features

- 🔍 **Fraud Detection**
  - Predicts whether a transaction is fraudulent or legitimate.
  - Uses trained machine learning models including Logistic Regression and Random Forest.

- 🌲 **Random Forest Prediction**
  - Uses the final Random Forest model for transaction classification.
  - Applies a configurable classification threshold to determine the final label.

- 📊 **SHAP Explainability**
  - Identifies the features that contributed most strongly to an individual prediction.
  - Shows whether each feature increased or decreased the model's fraud probability.
  - Provides both tabular and visual feature contributions.

- 🤖 **LLM-Powered Explanations**
  - Uses the Groq API with an OpenAI-compatible interface.
  - Converts SHAP contributions into concise, human-readable explanations.
  - The LLM explains the model's prediction and does not independently detect fraud.

- 🎲 **Sample Transactions**
  - Includes multiple predefined fraudulent and legitimate transactions.
  - Transactions are randomly selected from the sample set for demonstration.

- ✍️ **Manual Transaction Input**
  - Allows users to enter transaction features manually and analyze the resulting prediction.

- 🎨 **Interactive Streamlit Interface**
  - Displays prediction probability, classification threshold, SHAP contributions, charts, and AI-generated explanations in one interface.

- ☁️ **Cloud Deployment**
  - Deployed using Streamlit Cloud for public access.

---

## 🧠 How It Works

The system follows the pipeline below:

```text
Transaction Input
       │
       ▼
Data Preparation
       │
       ▼
Random Forest Model
       │
       ├──────────────► Fraud Probability
       │
       ▼
Classification Threshold
       │
       ▼
Fraud / Legitimate Prediction
       │
       ▼
SHAP Explainability
       │
       ▼
Top Feature Contributions
       │
       ▼
Groq LLM
       │
       ▼
Human-Readable Explanation

The machine learning model is responsible for the actual prediction.

SHAP is used to explain the contribution of individual features, while the LLM converts those technical contributions into a simpler explanation for the user.

📊 Dataset

The project uses the Credit Card Fraud Detection dataset containing anonymized transaction features.

The dataset contains:

Time — Time elapsed since the first transaction
V1–V28 — Anonymized PCA-transformed features
Amount — Transaction amount
Class — Target variable
0 → Legitimate
1 → Fraudulent

The V1–V28 features are anonymized PCA components. Their real-world meanings are not assumed or inferred by the application.

The original dataset is used for model development and evaluation, while a separate sample transaction JSON file is used by the deployed application for demonstration transactions.

🔬 Explainability with SHAP

The application uses SHAP to explain individual Random Forest predictions.

For each transaction, the system identifies the features with the strongest contributions.

Positive SHAP value → pushes the prediction toward fraud
Negative SHAP value → pushes the prediction away from fraud

The application displays:

Feature name
Feature value
SHAP contribution
Direction of influence
SHAP contribution chart

This makes the model's decision more transparent instead of providing only a final prediction.

🤖 LLM Explanation Layer

The project integrates a Groq-hosted LLM through an OpenAI-compatible API.

The LLM receives:

Random Forest prediction
Fraud probability
Classification threshold
Important SHAP feature contributions

It then generates a short explanation in natural language.

Important Design Principle

The LLM does not make the fraud prediction.

The architecture is:

Random Forest
      │
      │ prediction
      ▼
SHAP
      │
      │ feature contributions
      ▼
Groq LLM
      │
      │ explanation
      ▼
User

This keeps the ML prediction separate from the natural-language explanation layer.

🖥️ Application

The Streamlit application provides three transaction input options.

1. Manual Transaction

Users can enter:

Time
Amount
V1–V28 feature values

The transaction is then passed through the prediction and explainability pipeline.

2. Sample Fraud Transaction

The application randomly selects a predefined fraudulent transaction from the sample transaction dataset.

3. Sample Legitimate Transaction

The application randomly selects a predefined legitimate transaction from the sample transaction dataset.

The sample transactions allow users to explore the application without manually entering all 30 features.

🛠️ Tech Stack
Category	Technologies
Language	Python
Data Processing	Pandas, NumPy
Machine Learning	Scikit-learn
Models	Logistic Regression, Random Forest
Explainability	SHAP
Generative AI	Groq API
Frontend / Application	Streamlit
Visualization	Plotly
Version Control	Git, GitHub
Deployment	Streamlit Cloud
📁 Project Structure
credit-card-fraud-ai/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   └── creditcard.csv
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
⚙️ Local Setup
1. Clone the repository
git clone https://github.com/Saumyasingh1205/credit-card-fraud-ai.git
cd credit-card-fraud-ai
2. Create a virtual environment
python -m venv venv
3. Activate the virtual environment
Windows PowerShell
.\venv\Scripts\activate
macOS / Linux
source venv/bin/activate
4. Install dependencies
pip install -r requirements.txt
5. Configure the Groq API key

The LLM explanation module requires a Groq API key.

For local Streamlit execution, create:

.streamlit/secrets.toml

and add:

GROQ_API_KEY = "your_groq_api_key"

Do not commit your API key to GitHub.

▶️ Run the Application

From the project root:

python -m streamlit run app/app.py

The application will open in your browser.

🔐 Environment & Secrets

The project uses Streamlit secrets for API credentials.

Required secret:

GROQ_API_KEY

For cloud deployment, the API key should be configured through the deployment platform's secrets management instead of being stored in the repository.

📈 Prediction Output

For every analyzed transaction, the application displays:

Prediction label
Fraud probability
Classification threshold
SHAP feature contributions
SHAP visualization
AI-generated explanation

Example:

Prediction: Legitimate
Fraud Probability: 0%
Threshold: 0.35

The final classification is determined by the Random Forest model and configured threshold.

🧪 Testing

The project includes testing for the LLM explanation module.

Run the test module using:

python tests/test_llm.py

The application can also be tested manually through the Streamlit interface using both sample transaction types and manually entered transactions.

🌐 Deployment

The application is deployed using Streamlit Cloud.

Updates pushed to the configured GitHub branch can trigger a new deployment.

Live Application

👉 https://credit-card-fraud-ai.streamlit.app/

🔮 Future Improvements

Potential future improvements include:

Improve the manual transaction input experience
Add automatic validation for transaction feature values
Add more representative sample transactions
Improve model calibration and probability interpretation
Add additional model comparison visualizations
Expand automated testing
Improve monitoring and logging
Add richer interactive SHAP visualizations
Improve the UI/UX based on user feedback
👥 Collaboration

This project was developed collaboratively using Git and GitHub.

The repository uses feature-based development and Git branches to organize different parts of the machine learning and application pipeline.

Major areas of development include:

Data analysis and preprocessing
Machine learning models
Random Forest pipeline
Model evaluation
SHAP explainability
LLM-based explanations
Streamlit application
Deployment

## 👥 Contributors

- **Aryamann Srivastava** — [GitHub](https://github.com/Aryamann687)
- **Saumya Singh** — [GitHub](https://github.com/Saumyasingh1205)

This project was developed collaboratively using Git and GitHub.

⚠️ Disclaimer

This project is intended for educational and demonstration purposes.

It should not be used as a production financial fraud detection system without additional validation, security controls, monitoring, model calibration, and domain-specific evaluation.

The anonymized dataset features (V1–V28) do not have publicly interpretable real-world meanings.

📄 License

This project is intended for educational and portfolio purposes.
