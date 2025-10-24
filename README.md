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
