
"""Compliance Copilot — evidence-based transaction case review and PDF export."""

from datetime import datetime
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo
import html
import pickle
import sys

import numpy as np
import pandas as pd
import streamlit as st

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[2]
APP_DIR = ROOT / "app"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from components.theme import apply_theme

st.set_page_config(
    page_title="Compliance Copilot | PaySim",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "errorBalanceOrig",
    "errorBalanceDest",
    "type_TRANSFER",
]

DATA_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
]

BERLIN_TZ = ZoneInfo("Europe/Berlin")

STATUS_OPTIONS = [
    "Draft · awaiting human review",
    "Needs additional evidence",
    "Reviewed · demo only",
]


@st.cache_resource
def load_model():
    paths = [
        ROOT / "xgboost_fraud_model.pkl",
        ROOT / "models" / "xgboost_fraud_model.pkl",
    ]

    for path in paths:
        if path.is_file():
            with path.open("rb") as file:
                model = pickle.load(file)

            if not hasattr(model, "predict_proba"):
                raise TypeError("Saved model lacks predict_proba.")

            names = list(getattr(model, "feature_names_in_", FEATURES))

            if names != FEATURES:
                raise ValueError(
                    f"Model feature order mismatch: {names}"
                )

            return model

    raise FileNotFoundError(
        "xgboost_fraud_model.pkl not found."
    )


@st.cache_data(show_spinner="Loading transaction data…")
def load_data():
    paths = (
        sorted((ROOT / "data").glob("*.csv"))
        or sorted((ROOT / "data" / "raw").glob("*.csv"))
    )

    if not paths:
        raise FileNotFoundError(
            "PaySim CSV not found in data/ or data/raw/."
        )

    return pd.read_csv(
        paths[0],
        usecols=DATA_COLUMNS,
        nrows=200_000,
    )


def make_features(row):
    amount = float(row["amount"])
    old_orig = float(row["oldbalanceOrg"])
    new_orig = float(row["newbalanceOrig"])
    old_dest = float(row["oldbalanceDest"])
    new_dest = float(row["newbalanceDest"])

    values = {
        "step": int(row["step"]),
        "amount": amount,
        "oldbalanceOrg": old_orig,
        "newbalanceOrig": new_orig,
        "oldbalanceDest": old_dest,
        "newbalanceDest": new_dest,
        "errorBalanceOrig": old_orig - amount - new_orig,
        "errorBalanceDest": old_dest + amount - new_dest,
        "type_TRANSFER": int(row["type"] == "TRANSFER"),
    }

    return pd.DataFrame([values], columns=FEATURES)


@st.cache_resource(show_spinner="Initializing model explanations…")
def get_explainer(_model):
    import shap

    return shap.TreeExplainer(
        _model,
        model_output="raw",
    )


def explain(model, X):
    import xgboost as xgb

    explanation = get_explainer(model)(X)

    values = np.asarray(explanation.values)
    base = np.asarray(explanation.base_values)

    if values.shape == (1, len(FEATURES)):
        impacts = values[0]
    elif values.shape == (1, len(FEATURES), 2):
        impacts = values[0, :, 1]
    else:
        raise ValueError(
            f"Unexpected SHAP shape: {values.shape}"
        )

    if base.size == 1:
        baseline = float(base.reshape(-1)[0])
    elif base.size == 2:
        baseline = float(base.reshape(-1)[1])
    else:
        raise ValueError(
            f"Unexpected SHAP baseline shape: {base.shape}"
        )

    margin = float(
        model.get_booster().predict(
            xgb.DMatrix(X, feature_names=FEATURES),
            output_margin=True,
        )[0]
    )

    difference = abs(
        baseline + float(np.sum(impacts)) - margin
    )

    if difference > 1e-3:
        raise ValueError(
            f"SHAP additivity validation failed: {difference:.6f}"
        )

    order = np.argsort(np.abs(impacts))[::-1]

    return [
        (FEATURES[i], float(impacts[i]))
        for i in order[:4]
    ]


def review_recommendation(score, status):
    if status == "Reviewed · demo only":
        return (
            "The demonstration review has been marked as reviewed. "
            "This status records an analyst annotation only and "
            "does not constitute a compliance approval, regulatory "
            "decision, or completed investigation."
        )

    if status == "Needs additional evidence":
        return (
            "Additional supporting evidence is required before "
            "the case can be assessed further. Review transaction "
            "context, account history, counterparty information, "
            "and any available supporting records."
        )

    if score >= 0.5:
        return (
            "The model output exceeds the demonstration "
            "classification threshold of 0.50. Prioritize this "
            "case for human review and verify the supporting "
            "transaction evidence before considering escalation. "
            "The model score alone does not establish fraud."
        )

    return (
        "The model output is below the demonstration "
        "classification threshold of 0.50. No model-based "
        "escalation is suggested by this threshold alone. "
        "Additional evidence or business rules may still "
        "justify an analyst review."
    )


def narrative(row, idx, score, drivers, analyst_notes, status):
    transaction_id = f"TX-{idx:07d}"

    positive = [
        (name, value)
        for name, value in drivers
        if value > 0
    ]

    negative = [
        (name, value)
        for name, value in drivers
        if value < 0
    ]

    increases = (
        ", ".join(
            f"{name} ({value:+.2f})"
            for name, value in positive
        )
        or "none among the leading factors"
    )

    decreases = (
        ", ".join(
            f"{name} ({value:+.2f})"
            for name, value in negative
        )
        or "none among the leading factors"
    )

    dataset_label = (
        "Fraud"
        if int(row["isFraud"]) == 1
        else "Non-fraud"
    )

    text = (
        "CASE SUMMARY\n"
        f"Case {transaction_id} concerns a {row['type']} "
        f"transaction recorded in the PaySim dataset at "
        f"transaction step {int(row['step'])}. "
        f"The recorded amount is {float(row['amount']):,.2f} "
        f"dataset units. Sender ID: {row['nameOrig']}. "
        f"Receiver ID: {row['nameDest']}.\n\n"

        "MODEL ASSESSMENT\n"
        f"The XGBoost classifier produced a fraud-class "
        f"score of {score:.2%}. "
        f"The dataset label is {dataset_label}. "
        "The score is not a calibrated probability of "
        "financial crime.\n\n"

        "EXPLAINABILITY ANALYSIS\n"
        "Leading SHAP contributions increasing the "
        f"model output (raw log-odds): {increases}. "
        "Leading contributions decreasing the "
        f"model output: {decreases}. "
        "These contributions describe model behavior "
        "rather than establishing causation or intent.\n\n"

        "ANALYST RECOMMENDATION\n"
        f"{review_recommendation(score, status)}"
    )

    if analyst_notes.strip():
        text += (
            "\n\nANALYST NOTES\n"
            + analyst_notes.strip()
        )

    text += (
        "\n\nDATA SCOPE & MODEL LIMITATIONS\n"
        "This case uses synthetic PaySim transaction data "
        "and is intended for research and demonstration. "
        "The model includes post-transaction balances "
        "and engineered balance-error features that may "
        "introduce target leakage. The application does "
        "not perform live monitoring, payment blocking, "
        "customer verification, or regulatory filings. "
        "This document is not an official suspicious "
        "activity report."
    )

    return text


def pdf_bytes(report, idx, score, row, status):
    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=19 * mm,
        leftMargin=19 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="CaseTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=21,
            textColor=colors.HexColor("#182B3C"),
            spaceAfter=8,
        )
    )

    styles.add(
        ParagraphStyle(
            name="CaseSubtitle",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#587181"),
            spaceAfter=12,
        )
    )

    styles.add(
        ParagraphStyle(
            name="CaseSection",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#176B72"),
            spaceBefore=10,
            spaceAfter=4,
        )
    )

    styles.add(
        ParagraphStyle(
            name="CaseBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            spaceAfter=7,
            alignment=TA_LEFT,
        )
    )

    generated_at = datetime.now(BERLIN_TZ).strftime(
        "%Y-%m-%d %H:%M %Z"
    )

    story = [
        Paragraph(
            "TRANSACTION RISK / CASE REVIEW",
            styles["CaseTitle"],
        ),
        Paragraph(
            "Model-assisted assessment | Analyst review document",
            styles["CaseSubtitle"],
        ),
    ]

    metadata = [
        ["Case reference", f"TX-{idx:07d}"],
        ["Generated (Europe/Berlin)", generated_at],
        ["Model fraud-class score", f"{score:.2%} (uncalibrated)"],
        [
            "Dataset label",
            "Fraud" if int(row["isFraud"]) else "Non-fraud",
        ],
        ["Review status", status],
    ]

    table = Table(
        metadata,
        colWidths=[53 * mm, 110 * mm],
        hAlign="LEFT",
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.HexColor("#E8F0F2"),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor("#1D3341"),
                ),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                (
                    "LINEBELOW",
                    (0, 0),
                    (-1, -2),
                    0.4,
                    colors.HexColor("#C5D3DA"),
                ),
            ]
        )
    )

    story.extend([table, Spacer(1, 6 * mm)])

    section_names = {
        "CASE SUMMARY",
        "MODEL ASSESSMENT",
        "EXPLAINABILITY ANALYSIS",
        "ANALYST RECOMMENDATION",
        "ANALYST NOTES",
        "DATA SCOPE & MODEL LIMITATIONS",
    }

    for section in report.split("\n\n"):
        heading, separator, body = section.partition("\n")

        if separator and heading in section_names:
            content = [
                Paragraph(
                    html.escape(heading),
                    styles["CaseSection"],
                ),
                Paragraph(
                    html.escape(body).replace("\n", "<br/>"),
                    styles["CaseBody"],
                ),
            ]

            story.append(KeepTogether(content))
        else:
            story.append(
                Paragraph(
                    html.escape(section).replace("\n", "<br/>"),
                    styles["CaseBody"],
                )
            )

    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#657784"))

        canvas.drawString(
            19 * mm,
            12 * mm,
            "MODEL-ASSISTED CASE REVIEW | RESEARCH DATA",
        )

        canvas.drawRightString(
            190 * mm,
            12 * mm,
            f"Page {doc.page}",
        )

    document.build(
        story,
        onFirstPage=footer,
        onLaterPages=footer,
    )

    return buffer.getvalue()


# ---------------------------------------------------------
# PAGE STYLING
# ---------------------------------------------------------

st.markdown(
    """
<style>
.block-container {
    padding-top:1.05rem!important;
    padding-bottom:.9rem!important;
    max-width:1380px;
}

div[data-testid="stVerticalBlock"] {
    gap: .75rem;
}

.cc-eye {
    font-size:.67rem;
    letter-spacing:.15em;
    font-weight:750;
    color:#54D4BE;
    margin-bottom:3px;
}

.cc-title {
    font-size:1.65rem;
    font-weight:760;
    color:#F1F6FA;
    line-height:1.15;
}

.cc-sub {
    font-size:.78rem;
    color:#A9B8C8;
    margin:5px 0 12px;
}

.cc-label {
    font-size: .76rem;
    font-weight: 750;
    color: #D5E1EC;
    margin: 14px 0 9px;
    line-height: 1.4;
}

.cc-card {
    background:linear-gradient(140deg,#1C2938,#202E40);
    border:1px solid #334459;
    border-radius:13px;
    padding:13px 16px;
    min-height:105px;
}

.cc-small {
    font-size:.72rem;
    color:#A9B8C8;
}

.cc-big {
    font-size:1.1rem;
    font-weight:750;
    color:#F1F6FA;
    margin-top:5px;
    overflow-wrap:anywhere;
}

.cc-panel {
    background:#1C2938;
    border:1px solid #334459;
    border-radius:13px;
    padding:13px 17px;
}

.cc-row {
    display:flex;
    justify-content:space-between;
    gap:10px;
    padding:7px 0;
    border-bottom:1px solid #334459;
    font-size:.79rem;
}

.cc-row:last-child {
    border-bottom:none;
}

.cc-row span {
    color:#A9B8C8;
}

.cc-row strong {
    color:#F1F6FA;
    text-align:right;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="cc-eye">PAYSIM / COMPLIANCE WORKFLOW</div>'
    '<div class="cc-title">Compliance Copilot</div>'
    '<div class="cc-sub">Evidence-based case review · '
    'Explainable XGBoost · Analyst PDF report</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# DATA AND MODEL
# ---------------------------------------------------------

try:
    model = load_model()
    df = load_data()
except Exception as exc:
    st.error(
        f"Compliance Copilot initialization failed: {exc}"
    )
    st.stop()


previous = st.session_state.get(
    "selected_transaction_index"
)

if previous not in df.index:
    previous = int(df.index[0])


# ---------------------------------------------------------
# TRANSACTION SELECTION
# ---------------------------------------------------------

left_select, right_select = st.columns(
    [1, 2.5],
    gap="small",
)

with left_select:
    category = st.selectbox(
        "Transaction category",
        [
            "Fraud-labelled",
            "Non-fraud-labelled",
            "All transactions",
        ],
        index=(
            0
            if int(df.at[previous, "isFraud"]) == 1
            else 1
        ),
    )

if category == "Fraud-labelled":
    candidates = df.index[
        df["isFraud"].eq(1)
    ].tolist()

elif category == "Non-fraud-labelled":
    candidates = df.index[
        df["isFraud"].eq(0)
    ].tolist()

else:
    candidates = df.index.tolist()

if not candidates:
    st.warning("No transactions in the selected category.")
    st.stop()

if previous not in candidates:
    candidates = [previous] + candidates

choices = candidates[:500]

if previous not in choices:
    choices = [previous] + choices[:499]

with right_select:
    idx = st.selectbox(
        "Transaction",
        choices,
        index=(
            choices.index(previous)
            if previous in choices
            else 0
        ),
        format_func=lambda i: (
            f"TX-{i:07d} · "
            f"{df.at[i, 'type']} · "
            f"{df.at[i, 'amount']:,.2f}"
        ),
    )

st.session_state["selected_transaction_index"] = int(idx)

row = df.loc[idx]
X = make_features(row)


# ---------------------------------------------------------
# MODEL INFERENCE AND SHAP
# ---------------------------------------------------------

try:
    score = float(
        model.predict_proba(X)[0, 1]
    )

    drivers = explain(model, X)

except Exception as exc:
    st.error(
        f"Model explanation failed: "
        f"{type(exc).__name__}: {exc}"
    )
    st.stop()


# ---------------------------------------------------------
# CASE SUMMARY
# ---------------------------------------------------------

st.markdown(
    '<div class="cc-label">CASE BRIEF · MODEL EVIDENCE</div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(
    4,
    gap="small",
)

cards = [
    (
        "CASE REFERENCE",
        f"TX-{idx:07d}",
        "Transaction record",
    ),
    (
        "TRANSACTION",
        str(row["type"]),
        f"Amount: {float(row['amount']):,.2f} dataset units",
    ),
    (
        "MODEL OUTPUT",
        f"{score:.2%}",
        "Fraud-class score · uncalibrated",
    ),
    (
        "DATASET LABEL",
        (
            "FRAUD"
            if int(row["isFraud"]) == 1
            else "NON-FRAUD"
        ),
        "Recorded dataset label",
    ),
]

for column, (label, value, subtitle) in zip(
    (c1, c2, c3, c4),
    cards,
):
    with column:
        st.markdown(
            '<div class="cc-card">'
            f'<div class="cc-small">{html.escape(label)}</div>'
            f'<div class="cc-big">{html.escape(value)}</div>'
            f'<div class="cc-small">{html.escape(subtitle)}</div>'
            '</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------
# EVIDENCE AND ANALYST REVIEW
# ---------------------------------------------------------

left, right = st.columns(
    [1, 1.2],
    gap="medium",
)

with left:
    st.markdown(
        '<div class="cc-label">EVIDENCE SNAPSHOT</div>',
        unsafe_allow_html=True,
    )

    items = [
        (
            "Transaction step",
            str(int(row["step"])),
        ),
        (
            "Sender ID",
            str(row["nameOrig"]),
        ),
        (
            "Receiver ID",
            str(row["nameDest"]),
        ),
        (
            "Leading SHAP driver",
            f"{drivers[0][0]} "
            f"({drivers[0][1]:+.2f} log-odds)",
        ),
        (
            "Model vs label",
            (
                "MATCH"
                if (score >= 0.5) == bool(row["isFraud"])
                else "DIFFER"
            ),
        ),
    ]

    st.markdown(
        '<div class="cc-panel">'
        + "".join(
            '<div class="cc-row">'
            f'<span>{html.escape(k)}</span>'
            f'<strong>{html.escape(v)}</strong>'
            '</div>'
            for k, v in items
        )
        + '</div>',
        unsafe_allow_html=True,
    )

with right:
    st.markdown(
        '<div class="cc-label">ANALYST REVIEW</div>',
        unsafe_allow_html=True,
    )

    status = st.selectbox(
        "Review status",
        STATUS_OPTIONS,
        key=f"review_status_{idx}",
    )

    notes = st.text_area(
        "Analyst notes (optional)",
        placeholder=(
            "Record observations, supporting evidence, "
            "or information gaps."
        ),
        height=110,
        key=f"compliance_notes_{idx}",
    )

    st.caption(
        "Review annotations are local to this application."
    )


# ---------------------------------------------------------
# REPORT PREVIEW
# ---------------------------------------------------------

report = narrative(
    row,
    idx,
    score,
    drivers,
    notes,
    status,
)

st.markdown(
    '<div class="cc-label">REPORT PREVIEW</div>',
    unsafe_allow_html=True,
)

with st.expander(
    "Read the generated case narrative",
    expanded=False,
):
    st.text(report)


# ---------------------------------------------------------
# PDF EXPORT
# ---------------------------------------------------------

try:
    pdf = pdf_bytes(
        report,
        idx,
        score,
        row,
        status,
    )

except Exception as exc:
    st.error(
        f"PDF generation failed: "
        f"{type(exc).__name__}: {exc}"
    )

else:
    st.download_button(
        "⬇ Download case-review PDF",
        data=pdf,
        file_name=f"PaySim_case_TX-{idx:07d}.pdf",
        mime="application/pdf",
        type="primary",
        width="stretch",
    )

st.caption(
    "PaySim Dataset · XGBoost Risk Assessment · "
    "Analyst Case Review"
)
