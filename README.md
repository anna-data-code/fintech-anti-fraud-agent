# 🛡️ PaySim Anti-Fraud & XAI Compliance Copilot

An enterprise-grade, end-to-end FinTech solution for real-time transaction fraud detection, Explainable AI (XAI) risk attribution, and automated regulatory compliance (SAR) reporting.

---

## 🚀 Executive Summary
As digital financial transactions scale, financial institutions face strict regulatory demands to not only block fraudulent transfers instantly but also to provide transparent, auditable explanations for every automated decision. 

**PaySim Anti-Fraud Copilot** bridges the gap between complex machine learning models and regulatory compliance by combining high-performance **XGBoost** classification, **SHAP** (SHapley Additive exPlanations) interpretability, and a modern **Streamlit** enterprise web application.

---

## ✨ Key Features

1. **📊 Executive Intelligence Dashboard:**
   - Real-time telemetry monitoring total transaction volume, flagged anomalies, and model precision metrics.
   - Interactive trend analysis for risk management teams.

2. **🕹️ Real-Time Transaction Risk Simulator:**
   - Interactive UI controls allowing analysts to simulate high-value transfers (`TRANSFER` and `CASH_OUT`) dynamically.
   - Instant calculation of core anomaly indicators (e.g., origin and destination balance discrepancies: `errorBalanceOrig`).

3. **🔍 Local & Global Explainability (XAI):**
   - Integration of SHAP TreeExplainer to demystify black-box predictions.
   - Interactive Plotly visualizations highlighting primary risk-driving features for individual transactions.

4. **🤖 GenAI Compliance Copilot & PDF Export:**
   - Automated conversion of complex model outputs and SHAP vectors into formal Suspicious Activity Report (SAR) narratives.
   - One-click PDF report generation and download via ReportLab.

---

## 🛠️ Technology Stack
* **Core Language:** Python 3.10+
* **Machine Learning:** XGBoost, Scikit-Learn, Pandas, NumPy
* **Explainable AI:** SHAP (SHapley Additive exPlanations)
* **Web Frontend & UI:** Streamlit, Plotly
* **Reporting & Compliance:** ReportLab (PDF Generation)
* **Version Control:** Git, GitHub

---

## 📂 Project Architecture
```text
Final_Project_FinTech_Anti_Fraud/
│
├── app.py                  # Main Streamlit enterprise dashboard (Multi-tab UI & Sidebar navigation)
├── utils.py                # Core logic, feature engineering, Plotly XAI charts, and SAR/PDF generators
├── xgboost_fraud_model.pkl # Pre-trained XGBoost classification model
├── X_test.pkl              # Test dataset sample for evaluation
└── README.md               # Project documentation
```

---

## ⚙️ Installation & Quickstart

Follow these steps to run the application locally on your machine:
1. Clone the repository:
```
git clone [https://github.com/your-username/paysim-anti-fraud-copilot.git](https://github.com/your-username/paysim-anti-fraud-copilot.git)
cd paysim-anti-fraud-copilot
```
2. Install dependencies:
```
pip install streamlit xgboost shap pandas numpy plotly reportlab joblib
```
3. Run the Streamlit application:
```
streamlit run app.py
```
4. Access the Dashboard:
Open your browser and navigate to http://localhost:8501.

---

## 📈 Preview of the Application

```
Executive Overview: High-level macroeconomic fraud tracking.
Simulator: Live slider inputs calculating anomaly risk in milliseconds.
Compliance Copilot: Instant Markdown SAR logs and downloadable audit-ready PDF documents.
```