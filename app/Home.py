
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="Fraud Intelligence | PaySim",
    page_icon="🛡️",
    layout="wide",
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

COLORS = {
    "background": "#101923",
    "surface": "#182632",
    "text": "#F1F5F9",
    "muted": "#9FB0BD",
    "teal": "#35C7B0",
    "red": "#D95C63",
    "amber": "#E5B85C",
}


# -------------------------------------------------
# CUSTOM STYLING
# -------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #101923;
        color: #F1F5F9;
    }

    [data-testid="stMetric"] {
        background: #182632;
        border: 1px solid #263B48;
        border-radius: 14px;
        padding: 20px;
    }

    [data-testid="stMetricLabel"] {
        color: #9FB0BD;
    }

    h1, h2, h3 {
        color: #F1F5F9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------
# LOAD REAL DATA
# -------------------------------------------------

@st.cache_data(show_spinner="Loading PaySim transactions...")
def load_data():
    files = list(DATA_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No PaySim CSV found in {DATA_DIR}"
        )

    return pd.read_csv(
        files[0],
        usecols=["step", "type", "amount", "isFraud"],
    )


df = load_data()


# -------------------------------------------------
# HEADER
# -------------------------------------------------

st.title("Financial Crime Intelligence")
st.caption(
    "PaySim transaction analytics | "
    "Historical synthetic dataset"
)

st.divider()


# -------------------------------------------------
# KPI CARDS
# -------------------------------------------------

total_transactions = len(df)
fraud_cases = int(df["isFraud"].sum())
fraud_rate = df["isFraud"].mean() * 100
fraud_amount = df.loc[df["isFraud"] == 1, "amount"].sum()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Transactions", f"{total_transactions:,}")
c2.metric("Fraud Cases", f"{fraud_cases:,}")
c3.metric("Fraud Rate", f"{fraud_rate:.4f}%")
c4.metric("Fraud Amount", f"${fraud_amount / 1e6:,.1f}M")

st.divider()


# -------------------------------------------------
# TRANSACTION ACTIVITY OVER TIME
# -------------------------------------------------

st.subheader("Transaction Activity Over Time")

activity = (
    df.groupby("step")
    .agg(
        transactions=("isFraud", "size"),
        fraud_cases=("isFraud", "sum"),
    )
    .reset_index()
)

fig_activity = px.line(
    activity,
    x="step",
    y="transactions",
    title="Transaction Volume by Simulation Hour",
)

fig_activity.update_traces(
    line_color=COLORS["teal"]
)

fig_activity.update_layout(
    template="plotly_dark",
    paper_bgcolor=COLORS["surface"],
    plot_bgcolor=COLORS["surface"],
    font_color=COLORS["text"],
    xaxis_title="Simulation Hour",
    yaxis_title="Transactions",
)

st.plotly_chart(
    fig_activity,
    use_container_width=True,
)


# -------------------------------------------------
# FRAUD BY TRANSACTION TYPE
# -------------------------------------------------

left, right = st.columns(2)

fraud_by_type = (
    df.groupby("type", as_index=False)
    .agg(
        total=("isFraud", "size"),
        fraud=("isFraud", "sum"),
    )
)

fraud_by_type["fraud_rate"] = (
    fraud_by_type["fraud"]
    / fraud_by_type["total"]
    * 100
)

with left:
    st.subheader("Fraud Cases by Type")

    fig_fraud = px.bar(
        fraud_by_type,
        x="type",
        y="fraud",
        color_discrete_sequence=[COLORS["red"]],
    )

    fig_fraud.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface"],
        font_color=COLORS["text"],
        xaxis_title="Transaction Type",
        yaxis_title="Fraud Cases",
    )

    st.plotly_chart(
        fig_fraud,
        use_container_width=True,
    )

with right:
    st.subheader("Fraud Rate by Type")

    fig_rate = px.bar(
        fraud_by_type,
        x="type",
        y="fraud_rate",
        color_discrete_sequence=[COLORS["amber"]],
    )

    fig_rate.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface"],
        font_color=COLORS["text"],
        xaxis_title="Transaction Type",
        yaxis_title="Fraud Rate (%)",
    )

    st.plotly_chart(
        fig_rate,
        use_container_width=True,
    )


# -------------------------------------------------
# DATASET TRANSPARENCY
# -------------------------------------------------

st.divider()

st.info(
    "Data source: PaySim synthetic financial transactions. "
    "Metrics describe historical simulated data, "
    "not live banking activity."
)
