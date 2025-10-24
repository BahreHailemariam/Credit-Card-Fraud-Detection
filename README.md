# 💳 Credit Card Fraud Detection

## 🧩 Project Overview
This project focuses on identifying fraudulent credit card transactions using machine learning.
The dataset contains anonymized transaction features with labels indicating whether a transaction is legitimate or fraudulent.
The main goal is to build, evaluate, and deploy a model that detects fraud with high precision while minimizing false positives.

## 🎯 Business Problem
Credit card fraud is a critical issue for financial institutions, resulting in millions of dollars in annual losses.
Stakeholders want to identify suspicious transactions in real-time while maintaining a smooth user experience.

**Key Objectives:**
- Predict fraudulent transactions with high accuracy
- Minimize false positives to reduce customer inconvenience
- Provide explainable insights to fraud analysts

## 🧠 Machine Learning Workflow
### 1️⃣ Data Understanding
- Source: Kaggle Credit Card Fraud Detection Dataset
- Features: 28 anonymized principal components (V1–V28), Time, Amount, and Class (target)
- Imbalanced dataset — only ~0.17% transactions are frauds

### 2️⃣ Data Preparation (Python)
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("creditcard.csv")

scaler = StandardScaler()
df["scaled_amount"] = scaler.fit_transform(df["Amount"].values.reshape(-1, 1))
df["scaled_time"] = scaler.fit_transform(df["Time"].values.reshape(-1, 1))

df = df.drop(["Time", "Amount"], axis=1)

X = df.drop("Class", axis=1)
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
```

### 3️⃣ Model Training
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
```

### 4️⃣ Model Evaluation
Metrics used:
- Precision
- Recall
- F1 Score
- AUC-ROC

```python
from sklearn.metrics import roc_auc_score, roc_curve
auc = roc_auc_score(y_test, y_pred)
print(f"AUC Score: {auc:.4f}")
```

### 5️⃣ Model Improvement
- Addressed imbalance using SMOTE or class weights
- Tested models: Logistic Regression, Random Forest, XGBoost
- Performed hyperparameter tuning with GridSearchCV

## 📊 Power BI Dashboard Insights
### 1️⃣ Fraud Detection Summary
- KPIs: Total Transactions, Fraud Transactions %, Precision, Recall
- Chart: Fraudulent vs Legitimate Transactions (Bar Chart)

### 2️⃣ Transaction Pattern Analysis
- Distribution of transaction amount by fraud status
- Fraud by time of day (line chart)
- Top merchants/users flagged

### 3️⃣ Model Performance Overview
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve

### 4️⃣ Risk Segmentation
- Heatmap showing high-risk customer segments
- Drill-through report: Transaction-level details for flagged users

## ⚙️ Tech Stack
| Tool | Purpose |
|------|----------|
| Python (pandas, scikit-learn, imbalanced-learn) | Data cleaning & modeling |
| Power BI / Tableau | Visualization & dashboard |
| SQL | Querying and aggregating transaction data |
| Streamlit / Flask | Model deployment |

## 🚀 Deployment
Deployed via Streamlit app:
```bash
streamlit run fraud_app.py
```

## 🕒 Automation & Maintenance
- ETL scripts scheduled with Airflow for daily data refresh
- Automated retraining pipeline triggered monthly
- Email alerts for performance drift or high fraud spikes

