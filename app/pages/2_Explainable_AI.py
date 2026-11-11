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
.xai-card {
    background: linear-gradient(140deg,#1C2938,#202E40);
    border: 1px solid #334459;
    border-radius: 12px;
    padding: 12px 15px;
    font-size: .84rem;
    line-height: 1.4;
}
.xai-muted {font-size:.74rem;color:#A9B8C8;}
.xai-value {font-size:1.6rem;font-weight:800;color:#F1F6FA;margin:4px 0;}
.xai-note {font-size:.72rem;color:#A9B8C8;margin-top:8px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="xai-eyebrow">PAYSIM / MODEL TRANSPARENCY</div><div class="xai-title">Explainable AI</div><div class="xai-sub">Transaction-level SHAP explanations · XGBoost Fraud Detection Model</div>', unsafe_allow_html=True)

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


# ============================================================
# SHAP FEATURE IMPACT & MODEL EXPLANATION
# ============================================================

st.markdown("""
<style>
.xai-impact-title {
    color: #D5E1EC;
    font-size: .83rem;
    font-weight: 750;
    letter-spacing: .025em;
    margin: 9px 0 11px;
}

.xai-summary-card {
    background: linear-gradient(140deg, #1C2938, #202E40);
    border: 1px solid #334459;
    border-radius: 12px;
    padding: 11px 14px;
    margin-bottom: 6px;
}

.xai-summary-label {
    color: #A9B8C8;
    font-size: .76rem;
    margin-bottom: 6px;
}

.xai-summary-value {
    color: #F1F6FA;
    font-size: 1.35rem;
    font-weight: 800;
}

.xai-driver {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    padding: 6px 0;
    border-bottom: 1px solid #334459;
    font-size: .82rem;
}

.xai-driver:last-child {
    border-bottom: 0;
}

.xai-driver-name {
    color: #D5E1EC;
    font-weight: 650;
    overflow-wrap: anywhere;
}

.xai-driver-value {
    font-weight: 750;
    white-space: nowrap;
}

.xai-explanation {
    color: #A9B8C8;
    font-size: .78rem;
    line-height: 1.55;
    margin-top: 11px;
}

@media (max-width: 900px) {
    .xai-summary-card {
        padding: 13px;
    }
}
</style>
""", unsafe_allow_html=True)


def impact_color(value):
    return "#F17C94" if value > 0 else "#54D4BE"


def driver_row(feature, value):
    color = impact_color(value)
    return (
        '<div class="xai-driver">'
        f'<span class="xai-driver-name">{html.escape(str(feature))}</span>'
        f'<span class="xai-driver-value" style="color:{color}">'
        f'{value:+.3f}</span>'
        '</div>'
    )


# SHAP values are already calculated above.
# Sort by absolute impact, without changing the values.
impact_df = ranked.copy()

impact_df["Absolute impact"] = (
    impact_df["SHAP (log-odds)"].abs()
)

impact_df = impact_df.sort_values(
    "Absolute impact",
    ascending=True,
)

# Largest absolute impact appears at the top of the chart.
chart_features = impact_df["Feature"].astype(str).tolist()
chart_values = impact_df["SHAP (log-odds)"].astype(float).tolist()

bar_colors = [
    impact_color(value)
    for value in chart_values
]

left, right = st.columns([1.85, 1], gap="medium")

# ============================================================
# LEFT — FEATURE IMPACT CHART
# ============================================================

with left:
    st.markdown(
        '<div class="xai-impact-title">'
        'SHAP FEATURE IMPACT · TRANSACTION LEVEL'
        '</div>',
        unsafe_allow_html=True,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=chart_values,
            y=chart_features,
            orientation="h",
            marker=dict(
                color=bar_colors,
                line=dict(width=0),
            ),
            customdata=np.abs(chart_values),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "SHAP contribution: %{x:+.4f} log-odds"
                "<extra></extra>"
            ),
        )
    )

    max_abs = max(
        float(np.max(np.abs(chart_values))),
        0.1,
    )

    fig.update_layout(
        height=320,
        margin=dict(
            l=10,
            r=20,
            t=15,
            b=42,
        ),
        paper_bgcolor="#1C2938",
        plot_bgcolor="#1C2938",
        font=dict(
            color="#D5E1EC",
            size=11,
        ),
        bargap=0.32,
        showlegend=False,
        xaxis=dict(
            title=dict(
                text="SHAP contribution · log-odds",
                font=dict(size=11),
            ),
            range=[-max_abs * 1.15, max_abs * 1.15],
            gridcolor="#334459",
            zeroline=True,
            zerolinecolor="#8093A8",
            zerolinewidth=1.5,
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            title=None,
            tickfont=dict(size=11),
            automargin=True,
            showgrid=False,
        ),
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config={"displayModeBar": False},
    )

    st.markdown(
        """
        <div class="xai-note">
          <span style="color:#F17C94">●</span>
          Pink increases the model fraud score.
          &nbsp;&nbsp;
          <span style="color:#54D4BE">●</span>
          Teal decreases it.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# RIGHT — MODEL EXPLANATION SUMMARY
# ============================================================

with right:
    st.markdown(
        '<div class="xai-impact-title">'
        'MODEL EXPLANATION SUMMARY'
        '</div>',
        unsafe_allow_html=True,
    )

    score_pct = probability * 100

    score_color = (
        "#F17C94"
        if probability >= 0.5
        else "#54D4BE"
    )

    st.markdown(
        f"""
        <div class="xai-summary-card">
          <div class="xai-summary-label">
            XGBoost fraud-class score · uncalibrated
          </div>
          <div class="xai-summary-value"
               style="color:{score_color}">
            {score_pct:.2f}%
          </div>
          <div class="xai-explanation">
            Transaction TX-{index:07d}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_positive = (
        ranked[ranked["SHAP (log-odds)"] > 0]
        .head(3)
    )

    top_negative = (
        ranked[ranked["SHAP (log-odds)"] < 0]
        .head(3)
    )

    st.markdown(
        '<div class="xai-summary-card">'
        '<div class="xai-summary-label">'
        'INCREASES FRAUD SCORE'
        '</div>'
        + (
            "".join(
                driver_row(
                    item["Feature"],
                    float(item["SHAP (log-odds)"]),
                )
                for _, item in top_positive.iterrows()
            )
            if not top_positive.empty
            else '<div class="xai-explanation">No positive contributions.</div>'
        )
        + '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="xai-summary-card">'
        '<div class="xai-summary-label">'
        'DECREASES FRAUD SCORE'
        '</div>'
        + (
            "".join(
                driver_row(
                    item["Feature"],
                    float(item["SHAP (log-odds)"]),
                )
                for _, item in top_negative.iterrows()
            )
            if not top_negative.empty
            else '<div class="xai-explanation">No negative contributions.</div>'
        )
        + '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# MODEL DIAGNOSTICS
# ============================================================

with st.expander(
    "Model Diagnostics · SHAP Validation",
    expanded=False,
):
    a, b, c = st.columns(3)

    a.metric(
        "Baseline margin",
        f"{baseline:+.4f}",
    )

    b.metric(
        "Reconstructed margin",
        f"{reconstructed:+.4f}",
    )

    c.metric(
        "Additivity difference",
        f"{consistency_gap:.6f}",
    )

    if consistency_gap > 1e-3:
        st.warning(
            "SHAP reconstruction differs from the model raw margin. "
            "Verify explainer/model compatibility."
        )

    st.dataframe(
        ranked,
        hide_index=True,
        width="stretch",
    )
