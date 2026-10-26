# 🛡️ Autonomous FinTech Anti-Fraud & Compliance Agent

An end-to-end Machine Learning and Explainable AI (XAI) solution built to detect real-time financial fraud and automate regulatory incident reporting for compliance teams.

---

## 📌 Project Overview

Financial institutions process millions of transactions daily, making manual fraud inspection impossible. This project implements an **XGBoost classification pipeline** optimized for extreme class imbalance, capable of flagging suspicious transactions in milliseconds and evaluating fraud risk with near-perfect accuracy.

---

## 📊 Week 1 Progress & Achievements

* **Exploratory Data Analysis (EDA):** Validated **6.36M** records. Identified that 100% of fraud instances occur exclusively within `TRANSFER` and `CASH_OUT` transaction types.
* **Smart Data Filtering:** Reduced dataset volume by 56% (down to 2.77M rows) by isolating risk-prone transaction types, accelerating ML processing speeds while retaining 100% of target cases.
* **Feature Engineering:** Formulated domain-specific balance discrepancy indicators (`errorBalanceOrig`, `errorBalanceDest`) to capture account-draining anomalies.
* **Baseline XGBoost Model:** Trained a scale-weighted XGBoost Classifier achieving **0.9991 ROC-AUC** and **99.76% Fraud Recall** on unseen test data.

---

## 📈 Model Performance Highlights

| Metric | Score | Business Impact |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.9991** | Near-perfect discrimination between genuine & fraud cases |
| **Recall (Fraud Class)** | **99.76%** | Caught 1,639 out of 1,643 fraudulent test transactions |
| **Precision (Fraud Class)** | **95.62%** | Minimal false alarms for legitimate users |
| **Inference Speed** | **2.3 seconds** | Processed 554,082 test cases in real-time |

---

## 📁 Repository Structure

```text
├── 01_eda_paysim.ipynb              # EDA, Feature Engineering & Baseline Model
├── paysim_week1_presentation.html   # Week 1 Interactive Executive Presentation
└── README.md                        # Project documentation
```

---

## ⚙️ How to Setup & Run

```
1. Prerequisites & Dataset
Download the original PaySim dataset from Kaggle:
Dataset Source: PaySim Synthetic Financial Datasets for Fraud Detection
Place the CSV file into a local /data/ folder in your environment.
```

2. Environment Setup
```
git clone https://github.com/anna-data-code/fintech-anti-fraud-agent.git
cd fintech-anti-fraud-agent
pip install pandas numpy scikit-learn xgboost
```
---

## 🚀 Roadmap

```
[x] Week 1: Data Exploration, Feature Engineering & XGBoost Baseline Model
[ ] Week 2: Model Fine-tuning, SHAP (Explainable AI) & Incident Report Logic
[ ] Week 3: Interactive Streamlit Cockpit & GenAI Compliance Agent Integration
```


