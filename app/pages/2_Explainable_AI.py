"""Explainable AI | Local SHAP explanation for a selected PaySim transaction."""
from pathlib import Path
import html
import pickle
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
from components.theme import apply_theme

st.set_page_config(page_title="Explainable AI | PaySim", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
apply_theme()

FEATURES = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "errorBalanceOrig", "errorBalanceDest", "type_TRANSFER"]
DATA_COLUMNS = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud"]

@st.cache_resource
def load_model():
    for path in (ROOT / "xgboost_fraud_model.pkl", ROOT / "models" / "xgboost_fraud_model.pkl"):
        if path.is_file():
            with path.open("rb") as f:
                model = pickle.load(f)  # Only load your own trusted model file.
            if not hasattr(model, "predict_proba"):
                raise TypeError("Saved model has no predict_proba method.")
            names = list(getattr(model, "feature_names_in_", FEATURES))
            if names != FEATURES:
                raise ValueError(f"Feature order mismatch: {names}")
            return model
    raise FileNotFoundError("Missing xgboost_fraud_model.pkl in project root or models/.")

@st.cache_data(show_spinner="Loading historical PaySim sample…")
def load_data():
    paths = sorted((ROOT / "data").glob("*.csv")) or sorted((ROOT / "data" / "raw").glob("*.csv"))
    if not paths:
        raise FileNotFoundError("PaySim CSV not found in data/ or data/raw/.")
    return pd.read_csv(paths[0], usecols=DATA_COLUMNS, nrows=200_000)

def build_features(row):
    amount = float(row["amount"])
    oo, no = float(row["oldbalanceOrg"]), float(row["newbalanceOrig"])
    od, nd = float(row["oldbalanceDest"]), float(row["newbalanceDest"])
    values = {
        "step": int(row["step"]), "amount": amount,
        "oldbalanceOrg": oo, "newbalanceOrig": no,
        "oldbalanceDest": od, "newbalanceDest": nd,
        "errorBalanceOrig": oo - amount - no,
        "errorBalanceDest": od + amount - nd,
        "type_TRANSFER": int(row["type"] == "TRANSFER"),
    }
    return pd.DataFrame([values], columns=FEATURES)

@st.cache_resource(show_spinner="Initializing SHAP explainer…")
def get_explainer(_model):
    import shap
    return shap.TreeExplainer(_model, model_output="raw")

def local_shap(explainer, X):
    explanation = explainer(X)
    values = np.asarray(explanation.values)
    base = np.asarray(explanation.base_values)
    # For binary XGBoost, raw SHAP values are normally shape (1, n_features).
    if values.ndim == 2 and values.shape == (1, len(FEATURES)):
        impacts = values[0]
    elif values.ndim == 3 and values.shape[:2] == (1, len(FEATURES)) and values.shape[2] == 2:
        impacts = values[0, :, 1]
    else:
        raise ValueError(f"Unexpected SHAP shape: {values.shape}")
    if base.size == 1:
        baseline = float(base.reshape(-1)[0])
    elif base.size == 2:
        baseline = float(base.reshape(-1)[1])
    else:
        raise ValueError(f"Unexpected SHAP baseline shape: {base.shape}")
    return impacts.astype(float), baseline

st.markdown("""
<style>
.block-container {padding-top:1.15rem !important;padding-bottom:1rem !important;max-width:1380px;}
div[data-testid="stVerticalBlock"] {gap:.55rem;}
.xai-eyebrow {font-size:.67rem;letter-spacing:.15em;font-weight:750;color:#54D4BE;margin:0 0 3px;}
.xai-title {font-size:1.65rem;line-height:1.15;font-weight:760;color:#F1F6FA;margin:0;}
.xai-sub {font-size:.78rem;color:#A9B8C8;margin:5px 0 13px;}
.xai-label {font-size:.77rem;font-weight:750;color:#D5E1EC;margin:5px 0 6px;}
.xai-card {background:linear-gradient(140deg,#1C2938,#202E40);border:1px solid #334459;border-radius:14px;padding:14px 17px;}
.xai-muted {font-size:.74rem;color:#A9B8C8;}
.xai-value {font-size:1.6rem;font-weight:800;color:#F1F6FA;margin:4px 0;}
.xai-note {font-size:.72rem;color:#A9B8C8;margin-top:8px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="xai-eyebrow">PAYSIM / MODEL TRANSPARENCY</div><div class="xai-title">Explainable AI</div><div class="xai-sub">Transaction-level SHAP explanation · XGBoost · Historical synthetic dataset</div>', unsafe_allow_html=True)

try:
    model, df = load_model(), load_data()
except Exception as exc:
    st.error(f"Initialization failed: {exc}")
    st.stop()

selected = st.session_state.get("selected_transaction_index")
if selected is None or selected not in df.index:
    selected = int(df.index[0])

left_select, right_select = st.columns([1, 2.5], gap="small")
with left_select:
    default_category = 0 if int(df.at[selected, "isFraud"]) == 1 else 1
    category = st.selectbox("Transaction category", ["Fraud-labelled", "Non-fraud-labelled", "All transactions"], index=default_category)

if category == "Fraud-labelled":
    candidates = df.index[df["isFraud"].eq(1)].tolist()
elif category == "Non-fraud-labelled":
    candidates = df.index[df["isFraud"].eq(0)].tolist()
else:
    candidates = df.index.tolist()

if not candidates:
    st.warning("No transactions in this category.")
    st.stop()

# Preserve the transaction selected on Risk Investigation, even when its category differs.
if selected not in candidates:
    category = "All transactions"
    candidates = df.index.tolist()
    st.info("Showing all transactions to retain the selection from Risk Investigation.")

# Keep dropdown bounded while ensuring the transferred transaction is present.
choices = candidates[:500]
if selected not in choices:
    choices = [selected] + choices[:499]
with right_select:
    index = st.selectbox(
        "Transaction",
        choices,
        index=choices.index(selected) if selected in choices else 0,
        format_func=lambda i: f"TX-{i:07d} · {df.at[i, 'type']} · {df.at[i, 'amount']:,.2f}",
    )
st.session_state["selected_transaction_index"] = int(index)
row = df.loc[index]
X = build_features(row)

try:
    probability = float(model.predict_proba(X)[0, 1])
    explainer = get_explainer(model)
    impacts, baseline = local_shap(explainer, X)
    reconstructed = baseline + float(impacts.sum())
    # SHAP raw contributions are in log-odds, not percentage points.
    raw_margin = float(model.get_booster().predict(
        __import__("xgboost").DMatrix(X, feature_names=FEATURES), output_margin=True
    )[0])
    consistency_gap = abs(reconstructed - raw_margin)
except ImportError:
    st.error("SHAP is not installed. Run: python -m pip install shap")
    st.stop()
except Exception as exc:
    st.error(f"SHAP explanation failed: {type(exc).__name__}: {exc}")
    st.stop()

# The investigation page already contains model scores and transaction details.
# This page is deliberately dedicated to interpreting model behavior.
rank_order = np.argsort(np.abs(impacts))[::-1]
ranked = pd.DataFrame({
    "Feature": np.array(FEATURES)[rank_order],
    "SHAP (log-odds)": impacts[rank_order],
    "Input value": X.iloc[0].to_numpy()[rank_order],
})

st.markdown('<div class="xai-label">WHY DID THE MODEL MAKE THIS PREDICTION?</div>', unsafe_allow_html=True)
st.caption(f"Explaining TX-{index:07d} · {html.escape(str(row['type']))} · All contributions are in raw log-odds, not percentage points")

left, right = st.columns([1.7, 1], gap="medium")
with left:
    st.markdown('<div class="xai-label">SHAP WATERFALL · FROM BASELINE TO PREDICTION</div>', unsafe_allow_html=True)
    # A waterfall communicates how each feature moves the model from its
    # baseline score to the selected transaction's output.
    order = np.argsort(np.abs(impacts))
    labels = ["Baseline"] + [FEATURES[i] for i in order] + ["Model output"]
    values = [baseline] + [float(impacts[i]) for i in order] + [0]
    measures = ["absolute"] + ["relative"] * len(order) + ["total"]
    fig = go.Figure(go.Waterfall(
        orientation="v", measure=measures, x=labels, y=values,
        increasing={"marker": {"color": "#F17C94"}},
        decreasing={"marker": {"color": "#54D4BE"}},
        totals={"marker": {"color": "#7C9CFA"}},
        connector={"line": {"color": "#546579", "width": 1}},
        hovertemplate="%{x}<br>%{y:+.4f} log-odds<extra></extra>",
    ))
    fig.update_layout(
        height=355, margin=dict(l=12, r=10, t=10, b=90),
        paper_bgcolor="#1C2938", plot_bgcolor="#1C2938",
        font=dict(color="#D5E1EC", size=10),
        xaxis=dict(tickangle=-38, showgrid=False),
        yaxis=dict(title="Model output (log-odds)", gridcolor="#334459"),
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

with right:
    st.markdown('<div class="xai-label">KEY DRIVERS</div>', unsafe_allow_html=True)
    positive = ranked[ranked["SHAP (log-odds)"] > 0].head(3)
    negative = ranked[ranked["SHAP (log-odds)"] < 0].head(3)
    st.markdown("**Increases fraud score**")
    if positive.empty:
        st.caption("No features increased the fraud score for this transaction.")
    else:
        for _, item in positive.iterrows():
            st.markdown(f'<div class="xai-card" style="padding:8px 12px;margin-bottom:6px;border-left:3px solid #F17C94"><strong>{html.escape(str(item["Feature"]))}</strong><span style="float:right;color:#F17C94">{item["SHAP (log-odds)"]:+.3f}</span></div>', unsafe_allow_html=True)
    st.markdown("**Decreases fraud score**")
    if negative.empty:
        st.caption("No features decreased the fraud score for this transaction.")
    else:
        for _, item in negative.iterrows():
            st.markdown(f'<div class="xai-card" style="padding:8px 12px;margin-bottom:6px;border-left:3px solid #54D4BE"><strong>{html.escape(str(item["Feature"]))}</strong><span style="float:right;color:#54D4BE">{item["SHAP (log-odds)"]:+.3f}</span></div>', unsafe_allow_html=True)

st.markdown('<div class="xai-note"><span style="color:#F17C94">●</span> Pink increases the model fraud score. &nbsp; <span style="color:#54D4BE">●</span> Teal decreases it. &nbsp; SHAP explains model behavior, not causality.</div>', unsafe_allow_html=True)

with st.expander("SHAP diagnostics · feature values · methodological limitations", expanded=False):
    a, b, c = st.columns(3)
    a.metric("Baseline margin", f"{baseline:+.4f}")
    b.metric("Reconstructed margin", f"{reconstructed:+.4f}")
    c.metric("Additivity difference", f"{consistency_gap:.6f}")
    if consistency_gap > 1e-3:
        st.warning("SHAP reconstruction differs from the model raw margin. Verify explainer/model compatibility.")
    st.dataframe(ranked, hide_index=True, width="stretch")
    st.warning("The model uses post-transaction balances and derived balance errors. These may introduce target leakage. Explanations show what the trained model learned; they are not evidence of causal fraud indicators or suitability for live decisions.")

st.caption("Research prototype · Synthetic PaySim data · Selection limited to first 200,000 rows · Not live monitoring")
