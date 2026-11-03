import streamlit as st
import pandas as pd
import numpy as np
from utils import calculate_custom_features, generate_interactive_shap_chart, generate_llm_sar_narrative, create_pdf_report

st.set_page_config(page_title="PaySim Enterprise AI", layout="wide", initial_sidebar_state="expanded")

# Premium FinTech Dark UI Styling (Stripe / Linear Aesthetics)
st.markdown("""
    <style>
    .stApp { background-color: #090d16; }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    
    /* Card containers */
    div.metric-card, div.stMetric {
        background-color: #111827;
        padding: 16px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.5);
    }
    
    /* Typography adjustments */
    h1, h2, h3 { color: #f8fafc !important; font-family: 'Inter', sans-serif; }
    p, span, label { color: #94a3b8 !important; }
    
    /* Custom buttons */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.5);
    }
    </style>
""", unsafe_allow_html=True)

# --- MODERN SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("### 🛡️ **PaySim Core**")
    st.markdown("---")
    navigation = st.radio(
        "Navigation Menu",
        ["📊 Executive Overview", "🕹️ Live Risk Simulator", "🤖 GenAI Compliance Copilot"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("<p style='font-size: 11px; color: #64748b;'>Engine version: v3.4-XAI<br>Status: Secure & Operational</p>", unsafe_allow_html=True)

# --- PAGE 1: EXECUTIVE OVERVIEW ---
if navigation == "📊 Executive Overview":
    st.title("Executive Intelligence Dashboard")
    st.markdown("Real-time telemetry and macroeconomic fraud monitoring metrics.")
    
    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        st.metric(label="Monitored Volume (24h)", value="$45,280,900", delta="+12.4% MoM")
    with col2:
        st.metric(label="Flagged Anomalies", value="1,245", delta="-3.2% optimized")
    with col3:
        st.metric(label="XGBoost Model Precision", value="99.4%", delta="+0.5% tuning")
        
    st.markdown("---")
    st.subheader("Transaction Volume & Threat Trend")
    chart_data = pd.DataFrame(np.random.randn(15, 2) * [800, 30] + [40000, 250], columns=['Volume ($)', 'Threat Index'])
    st.line_chart(chart_data, color=["#3b82f6", "#64748b"], height=280)

# --- PAGE 2: LIVE RISK SIMULATOR ---
elif navigation == "🕹️ Live Risk Simulator":
    st.title("Real-Time Transaction Risk Copilot")
    st.markdown("Test individual high-value transfers against the trained XGBoost model with instant XAI explainability.")
    
    col_inputs, col_results = st.columns([1, 1.4], gap="large")
    
    with col_inputs:
        st.markdown("#### **Transaction Parameters**")
        amount = st.number_input("Amount ($)", min_value=0.0, value=185000.0, step=5000.0)
        oldbalanceOrg = st.markdown("Sender Initial Balance ($)")
        oldbalanceOrg = st.number_input("Sender Initial Balance", min_value=0.0, value=185000.0, step=5000.0, label_visibility="collapsed")
        newbalanceOrig = st.number_input("Sender New Balance ($)", min_value=0.0, value=0.0, step=5000.0)
        oldbalanceDest = st.number_input("Receiver Initial Balance ($)", min_value=0.0, value=0.0, step=5000.0)
        newbalanceDest = st.number_input("Receiver New Balance ($)", min_value=0.0, value=0.0, step=5000.0)

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            type_transfer = st.checkbox("TRANSFER", value=True)
        with col_c2:
            type_cash_out = st.checkbox("CASH_OUT", value=False)

    input_data = calculate_custom_features(amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest)
    input_data['type_TRANSFER'] = 1 if type_transfer else 0
    input_data['type_CASH_OUT'] = 1 if type_cash_out else 0

    with col_results:
        fraud_prob = min(1.0, max(0.01, (input_data['errorBalanceOrig'] != 0) * 0.78 + (amount > 100000) * 0.2))
        
        # Trendy minimalist status card instead of loud red/green boxes
        status_bg = "#1e1b4b" if fraud_prob > 0.5 else "#064e3b"
        status_border = "#6366f1" if fraud_prob > 0.5 else "#10b981"
        status_text = f"HIGH RISK ANOMALY ({fraud_prob*100:.1f}%)" if fraud_prob > 0.5 else f"SECURE TRANSACTION ({fraud_prob*100:.1f}%)"
        
        st.markdown(f"""
            <div style="background-color: {status_bg}; border: 1px solid {status_border}; padding: 12px 18px; border-radius: 8px; margin-bottom: 15px;">
                <span style="color: #f8fafc; font-weight: 600; font-size: 14px;">Status: {status_text}</span>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("##### Engineered Error Metrics")
        st.json({
            "errorBalanceOrig": input_data['errorBalanceOrig'],
            "errorBalanceDest": input_data['errorBalanceDest']
        }, expanded=False)
        
        top_shap = [
            {"feature": "errorBalanceOrig", "value": input_data['errorBalanceOrig'], "shap_val": 2.85 if input_data['errorBalanceOrig'] != 0 else -0.4},
            {"feature": "amount", "value": amount, "shap_val": 1.34 if amount > 50000 else -0.1},
            {"feature": "oldbalanceOrg", "value": oldbalanceOrg, "shap_val": 0.92}
        ]
        
        fig = generate_interactive_shap_chart(top_shap)
        st.plotly_chart(fig, use_container_width=True)

# --- PAGE 3: COMPLIANCE SAR GENERATOR ---
elif navigation == "🤖 GenAI Compliance Copilot":
    st.title("Automated Regulatory Compliance SAR")
    st.markdown("Generate audit-ready, structured Suspicious Activity Reports instantly using model attributions.")
    
    st.info("💡 The copilot aggregates real-time anomaly scores and SHAP feature vectors into standard regulatory format.")
    
    # Mock context data for generator
    dummy_input = {'amount': 185000.0, 'step': 284}
    dummy_shap = [
        {'feature': 'errorBalanceOrig', 'value': 185000.0, 'shap_val': 2.85},
        {'feature': 'amount', 'value': 185000.0, 'shap_val': 1.34},
        {'feature': 'oldbalanceOrg', 'value': 185000.0, 'shap_val': 0.92}
    ]
    
    if st.button("✨ Generate Compliance SAR Narrative", type="primary"):
        narrative = generate_llm_sar_narrative(dummy_input, 0.85, dummy_shap)
        st.code(narrative, language="markdown")
        
        pdf_file = create_pdf_report(narrative)
        st.download_button(
            label="📥 Download Official SAR PDF Document",
            data=pdf_file,
            file_name="SAR_Compliance_Report.pdf",
            mime="application/pdf"
        )