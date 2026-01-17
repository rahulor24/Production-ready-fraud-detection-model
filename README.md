# 💳 Fraud Detection in Financial Transactions

This project builds an end-to-end machine learning pipeline to detect fraudulent financial transactions using real-world banking data. It includes exploratory analysis, feature engineering, model training, interpretability, and deployment via a Streamlit app.

## Click on the link below to see the final deployed interactive UI
[Streamlit UI](https://fraud-detection-supervised.streamlit.app/)

## 📁 Project Structure
.
-├── data/                     # Dataset (not committed if sensitive)
-│
-├── notebooks/                # Research & experimentation
-│   ├── 01_eda.ipynb           # Exploratory Data Analysis
-│   ├── 02_preprocessing.ipynb # Feature engineering & preprocessing
-│   ├── 03_baseline_modeling.ipynb
-│   ├── 04_advanced_modeling.ipynb
-│   └── catboost_info/         # CatBoost artifacts
-│
-├── src/                      # Production-ready code
-│   ├── main.py                # Model loading & inference logic
-│   ├── preprocessor.py        # Feature preprocessing pipeline
-│   ├── shap_rf.py             # SHAP explainability module (Random Forest)
-│   └── st.py                  # Streamlit application entry point
-│
-├── .gitignore
-├── requirements.txt           # Deployment-safe dependencies
-└── README.md

## 🧠 Problem Statement

Predict whether a financial transaction is fraudulent or legitimate based on features like amount, velocity, IP risk score, geo-distance, merchant risk, and more.

## ⚙️ Key Features

- Domain-specific feature engineering: `velocity`, `ip_risk_score`, `geo_distance`, `merchant_risk_score`
- Multiple models: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, AdaBoost, SVM
- Class imbalance handling: SMOTE, class weights
- Model interpretability: SHAP values, permutation importance
- Streamlit app for interactive fraud prediction

## 🚀 How to Run

1. Clone the repo  
2. Install dependencies: `pip install -r requirements.txt`  
3. Launch app: `streamlit run src/main.py`

## 📊 Sample Features

- `amount`, `currency`, `transaction_type`, `channel`
- `card_present`, `device_id`, `location`, `is_international`
- `failed_login_attempts`, `velocity`, `ip_risk_score`
- `customer_age`, `account_tenure`, `geo_distance`
- `merchant_risk_score`, `fraud_flag`

## 📈 Evaluation Metrics

- ROC-AUC, Precision-Recall, F1-score  
- Confusion matrix and SHAP plots for interpretability

## 🧪 Status

✅ EDA, preprocessing, modeling, and interpretability complete  
🚧 Deployment via Streamlit complete

## 📬 Contact

For questions or collaboration, feel free to reach out via LinkedIn or GitHub.
[LinkedIn](https://www.linkedin.com/in/rahuldu/)
