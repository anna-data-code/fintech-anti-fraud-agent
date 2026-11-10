"""Compliance Copilot: evidence-grounded, human-review SAR-style draft for PaySim."""
from pathlib import Path
from io import BytesIO
from datetime import datetime, timezone
import html
import pickle
import sys

import numpy as np
import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether

ROOT = Path(__file__).resolve().parents[2]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
from components.theme import apply_theme

st.set_page_config(page_title="Compliance Copilot | PaySim", page_icon="📋",
                   layout="wide", initial_sidebar_state="expanded")
apply_theme()

FEATURES = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest",
            "newbalanceDest", "errorBalanceOrig", "errorBalanceDest", "type_TRANSFER"]
DATA_COLUMNS = ["step", "type", "amount", "nameOrig", "oldbalanceOrg",
                "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud"]

@st.cache_resource
def load_model():
    for path in (ROOT / "xgboost_fraud_model.pkl", ROOT / "models" / "xgboost_fraud_model.pkl"):
        if path.is_file():
            with path.open("rb") as f:
                model = pickle.load(f)  # Trusted local artifact only.
            if not hasattr(model, "predict_proba"):
                raise TypeError("Saved model lacks predict_proba.")
            names = list(getattr(model, "feature_names_in_", FEATURES))
            if names != FEATURES:
                raise ValueError(f"Feature mismatch: {names}")
            return model
    raise FileNotFoundError("xgboost_fraud_model.pkl missing in root or models/.")

@st.cache_data(show_spinner="Loading PaySim transaction sample…")
def load_data():
    paths = sorted((ROOT / "data").glob("*.csv")) or sorted((ROOT / "data" / "raw").glob("*.csv"))
    if not paths:
        raise FileNotFoundError("PaySim CSV missing in data/ or data/raw/.")
    return pd.read_csv(paths[0], usecols=DATA_COLUMNS, nrows=200_000)

def make_features(row):
    amount = float(row["amount"])
    oo, no = float(row["oldbalanceOrg"]), float(row["newbalanceOrig"])
    od, nd = float(row["oldbalanceDest"]), float(row["newbalanceDest"])
    return pd.DataFrame([{
        "step": int(row["step"]), "amount": amount,
        "oldbalanceOrg": oo, "newbalanceOrig": no,
        "oldbalanceDest": od, "newbalanceDest": nd,
        "errorBalanceOrig": oo - amount - no,
        "errorBalanceDest": od + amount - nd,
        "type_TRANSFER": int(row["type"] == "TRANSFER"),
    }], columns=FEATURES)

@st.cache_resource(show_spinner="Loading SHAP explainer…")
def get_explainer(_model):
    import shap
    return shap.TreeExplainer(_model, model_output="raw")

def explain(model, X):
    exp = get_explainer(model)(X)
    vals = np.asarray(exp.values)
    if vals.shape == (1, len(FEATURES)):
        impacts = vals[0]
    elif vals.shape == (1, len(FEATURES), 2):
        impacts = vals[0, :, 1]
    else:
        raise ValueError(f"Unexpected SHAP shape: {vals.shape}")
    baseline_array = np.asarray(exp.base_values).reshape(-1)
    if baseline_array.size not in (1, 2):
        raise ValueError(f"Unexpected baseline shape: {baseline_array.shape}")
    baseline = float(baseline_array[-1])
    import xgboost as xgb
    margin = float(model.get_booster().predict(
        xgb.DMatrix(X, feature_names=FEATURES), output_margin=True)[0])
    if abs(baseline + float(np.sum(impacts)) - margin) > 1e-3:
        raise ValueError("SHAP additivity check failed.")
    order = np.argsort(np.abs(impacts))[::-1]
    return [(FEATURES[i], float(impacts[i])) for i in order[:4]]

def narrative(row, idx, score, drivers, analyst_notes):
    tx = f"TX-{idx:07d}"
    positives = [(name, value) for name, value in drivers if value > 0]
    negatives = [(name, value) for name, value in drivers if value < 0]
    up = ", ".join(f"{name} ({value:+.2f})" for name, value in positives) or "none among the leading factors"
    down = ", ".join(f"{name} ({value:+.2f})" for name, value in negatives) or "none among the leading factors"
    text = (
        f"CASE SUMMARY\n"
        f"Transaction {tx} is a historical, synthetic PaySim {row['type']} transaction "
        f"recorded at simulation hour {int(row['step'])}, with amount {float(row['amount']):,.2f} "
        f"(dataset units). The simulated sender identifier is {row['nameOrig']} and "
        f"the simulated receiver identifier is {row['nameDest']}.\n\n"
        f"MODEL OBSERVATION\n"
        f"The saved XGBoost classifier produced a fraud-class score of {score:.2%}. "
        f"The PaySim ground-truth label for this record is "
        f"{'fraud' if int(row['isFraud']) == 1 else 'non-fraud'}. "
        f"This is a retrospective model assessment, not a real-time alert or confirmed crime.\n\n"
        f"EXPLANATION\n"
        f"Leading SHAP factors increasing the model's fraud-class output (raw log-odds): {up}. "
        f"Leading factors decreasing it: {down}. These are model attributions, not proof of intent "
        f"or causation.\n\n"
        f"REVIEW RECOMMENDATION\n"
        f"Treat this as an illustrative analyst-review case only. Validate transaction context, "
        f"customer and counterparty information, supporting records, and relevant regulatory "
        f"requirements before deciding whether any real-world escalation or filing is warranted. "
        f"No hold, referral, investigation, or regulatory filing has been performed by this application."
    )
    if analyst_notes.strip():
        text += "\n\nANALYST NOTES (USER ENTERED)\n" + analyst_notes.strip()
    text += (
        "\n\nLIMITATIONS\n"
        "Synthetic historical data; no live monitoring. The model includes post-transaction balance "
        "fields and engineered balance errors that may introduce target leakage. Scores are not "
        "calibrated financial-risk probabilities. This document is not an official SAR filing."
    )
    return text

def pdf_bytes(report, idx, score, row):
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=19*mm,
                            leftMargin=19*mm, topMargin=18*mm, bottomMargin=18*mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitleCustom", parent=styles["Title"],
                              fontName="Helvetica-Bold", fontSize=16, leading=20,
                              textColor=colors.HexColor("#182B3C"), spaceAfter=8))
    styles.add(ParagraphStyle(name="ReportSectionCustom", parent=styles["Heading2"],
                              fontName="Helvetica-Bold", fontSize=10, leading=14,
                              textColor=colors.HexColor("#176B72"), spaceBefore=11, spaceAfter=4))
    styles.add(ParagraphStyle(name="ReportBodyCustom", parent=styles["BodyText"],
                              fontName="Helvetica", fontSize=9, leading=13, spaceAfter=7,
                              alignment=TA_LEFT))
    story = [
        Paragraph("COMPLIANCE REVIEW / DRAFT", styles["ReportTitleCustom"]),
        Paragraph("Synthetic PaySim research prototype · NOT AN OFFICIAL SAR",
                  styles["ReportBodyCustom"]),
    ]
    metadata = [
        ["Case reference", f"TX-{idx:07d}"],
        ["Generated (UTC)", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")],
        ["Model fraud-class score", f"{score:.2%} (uncalibrated)"],
        ["Dataset label", "Fraud" if int(row["isFraud"]) else "Non-fraud"],
        ["Status", "Draft / human review required"],
    ]
    table = Table(metadata, colWidths=[47*mm, 116*mm], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E8F0F2")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1D3341")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.HexColor("#C5D3DA")),
    ]))
    story.extend([table, Spacer(1, 7*mm)])
    sections = report.split("\n\n")
    for section in sections:
        first, _, rest = section.partition("\n")
        if first in {"CASE SUMMARY", "MODEL OBSERVATION", "EXPLANATION",
                     "REVIEW RECOMMENDATION", "ANALYST NOTES (USER ENTERED)", "LIMITATIONS"}:
            story.append(Paragraph(html.escape(first), styles["ReportSectionCustom"]))
            story.append(Paragraph(html.escape(rest).replace("\n", "<br/>"),
                                   styles["ReportBodyCustom"]))
        else:
            story.append(Paragraph(html.escape(section).replace("\n", "<br/>"),
                                   styles["ReportBodyCustom"]))
    def footer(canvas, document):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#657784"))
        canvas.drawString(19*mm, 12*mm, "RESEARCH DEMONSTRATION · NOT A REGULATORY SUBMISSION")
        canvas.drawRightString(190*mm, 12*mm, f"Page {document.page}")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()

st.markdown("""
<style>
.block-container {padding-top:1.05rem!important;padding-bottom:.9rem!important;max-width:1380px;}
div[data-testid="stVerticalBlock"] {gap:.52rem;}
.cc-eye {font-size:.67rem;letter-spacing:.15em;font-weight:750;color:#54D4BE;margin-bottom:3px}
.cc-title {font-size:1.65rem;font-weight:760;color:#F1F6FA;line-height:1.15}
.cc-sub {font-size:.78rem;color:#A9B8C8;margin:5px 0 12px}
.cc-label {font-size:.76rem;font-weight:750;color:#D5E1EC;margin:4px 0 5px}
.cc-card {background:linear-gradient(140deg,#1C2938,#202E40);border:1px solid #334459;
border-radius:13px;padding:13px 16px;min-height:105px}
.cc-small {font-size:.72rem;color:#A9B8C8}
.cc-big {font-size:1.1rem;font-weight:750;color:#F1F6FA;margin-top:5px;overflow-wrap:anywhere}
.cc-panel {background:#1C2938;border:1px solid #334459;border-radius:13px;padding:13px 17px}
.cc-row {display:flex;justify-content:space-between;gap:10px;padding:7px 0;
border-bottom:1px solid #334459;font-size:.79rem}
.cc-row:last-child {border-bottom:none}
.cc-row span {color:#A9B8C8}.cc-row strong{color:#F1F6FA;text-align:right}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="cc-eye">PAYSIM / COMPLIANCE WORKFLOW</div>'
            '<div class="cc-title">Compliance Copilot</div>'
            '<div class="cc-sub">Evidence-grounded case brief · Explainable XGBoost · Human-review PDF draft</div>',
            unsafe_allow_html=True)
try:
    model, df = load_model(), load_data()
except Exception as exc:
    st.error(f"Cannot initialize Compliance Copilot: {exc}")
    st.stop()

previous = st.session_state.get("selected_transaction_index")
if previous not in df.index:
    previous = int(df.index[0])

a, b = st.columns([1, 2.5], gap="small")
with a:
    category = st.selectbox("Transaction category",
                            ["Fraud-labelled", "Non-fraud-labelled", "All transactions"],
                            index=0 if int(df.at[previous, "isFraud"]) else 1)
if category == "Fraud-labelled":
    candidates = df.index[df["isFraud"].eq(1)].tolist()
elif category == "Non-fraud-labelled":
    candidates = df.index[df["isFraud"].eq(0)].tolist()
else:
    candidates = df.index.tolist()
if previous not in candidates:
    candidates = [previous] + candidates
choices = candidates[:500]
if previous not in choices:
    choices = [previous] + choices[:499]
with b:
    idx = st.selectbox("Transaction", choices,
                       index=choices.index(previous) if previous in choices else 0,
                       format_func=lambda i: f"TX-{i:07d} · {df.at[i, 'type']} · {df.at[i, 'amount']:,.2f}")
st.session_state["selected_transaction_index"] = int(idx)
row = df.loc[idx]
X = make_features(row)
try:
    score = float(model.predict_proba(X)[0, 1])
    drivers = explain(model, X)
except Exception as exc:
    st.error(f"Model/SHAP computation failed: {type(exc).__name__}: {exc}")
    st.stop()

st.markdown('<div class="cc-label">CASE BRIEF · VERIFIED DATA</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4, gap="small")
cards = [
    ("CASE REFERENCE", f"TX-{idx:07d}", "Historical synthetic record"),
    ("TRANSACTION", str(row["type"]), f"Amount: {float(row['amount']):,.2f} dataset units"),
    ("MODEL OUTPUT", f"{score:.2%}", "Fraud-class score · uncalibrated"),
    ("DATASET LABEL", "FRAUD" if int(row["isFraud"]) else "NON-FRAUD", "Ground truth in PaySim"),
]
for column, (label, value, subtitle) in zip((c1, c2, c3, c4), cards):
    with column:
        st.markdown(
            f'<div class="cc-card"><div class="cc-small">{html.escape(label)}</div>'
            f'<div class="cc-big">{html.escape(value)}</div>'
            f'<div class="cc-small">{html.escape(subtitle)}</div></div>',
            unsafe_allow_html=True)

left, right = st.columns([1, 1.2], gap="medium")
with left:
    st.markdown('<div class="cc-label">EVIDENCE SNAPSHOT</div>', unsafe_allow_html=True)
    items = [
        ("Simulation hour", str(int(row["step"]))),
        ("Sender (synthetic)", str(row["nameOrig"])),
        ("Receiver (synthetic)", str(row["nameDest"])),
        ("Leading SHAP driver", f"{drivers[0][0]} ({drivers[0][1]:+.2f} log-odds)"),
        ("Model vs label", "MATCH" if (score >= .5) == bool(row["isFraud"]) else "DIFFER"),
    ]
    st.markdown('<div class="cc-panel">' + "".join(
        f'<div class="cc-row"><span>{html.escape(k)}</span><strong>{html.escape(v)}</strong></div>'
        for k, v in items) + '</div>', unsafe_allow_html=True)
    with st.expander("Methodology and review safeguards", expanded=False):
        st.warning("Synthetic PaySim data. The trained model includes post-transaction balances and "
                   "balance-error features, which may leak target information. SHAP explains the model, "
                   "not fraud causation. No live alerts, transaction holds, or filings occur.")
        st.caption("No customer KYC records, sanctions screening, legal assessment, or real-world case evidence are available.")

with right:
    st.markdown('<div class="cc-label">ANALYST REVIEW</div>', unsafe_allow_html=True)
    status = st.selectbox("Review status", ["Draft · awaiting human review",
                                           "Needs additional evidence", "Reviewed · demo only"])
    notes = st.text_area("Analyst notes (optional)",
                         placeholder="Add observations or evidence gaps. Avoid unsupported allegations.",
                         height=110, key=f"compliance_notes_{idx}")
    st.caption("The review status is a local demo annotation; it does not trigger an external workflow.")

report = narrative(row, idx, score, drivers, notes)
st.markdown('<div class="cc-label">REPORT PREVIEW</div>', unsafe_allow_html=True)
with st.expander("Read the generated case narrative", expanded=False):
    st.text(report)
st.caption("Deterministic narrative built from the selected record, saved model and verified SHAP values. "
           "No LLM API is called; no regulatory SAR is filed.")

try:
    pdf = pdf_bytes(report + f"\n\nANALYST REVIEW STATUS\n{status}", idx, score, row)
except Exception as exc:
    st.error(f"PDF generation failed: {type(exc).__name__}: {exc}")
else:
    st.download_button("⬇ Download case-review PDF", data=pdf,
                       file_name=f"PaySim_case_TX-{idx:07d}.pdf", mime="application/pdf",
                       type="primary", use_container_width=True)
st.caption("Research prototype · First 200,000 PaySim rows · No live monitoring · Human review required")
