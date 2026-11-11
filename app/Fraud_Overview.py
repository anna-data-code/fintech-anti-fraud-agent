
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PaySim | Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

C = {
    "bg": "#11151D",
    "surface": "#1C2430",
    "border": "#303D4D",
    "text": "#F4F7FB",
    "muted": "#A0ADBD",
    "mint": "#54D6C1",
    "coral": "#F17888",
    "amber": "#F0BE6D",
}


# ============================================================
# STYLING
# ============================================================


st.markdown(
    """
    <style>
    .stApp {
        background: #11151D;
        color: #F4F7FB;
    }

    .block-container {
        max-width: 1450px;
        padding-top: clamp(0.55rem, 1.5vh, 1.1rem);
        padding-bottom: 0.9rem;
    }

    .fi-eyebrow {
        color: #54D6C1;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.4px;
        margin-bottom: 3px;
    }

    .fi-title {
        color: #F4F7FB;
        font-size: clamp(21px, 2vw, 27px);
        font-weight: 750;
        line-height: 1.15;
        margin: 0 0 3px;
    }

    .fi-subtitle {
        color: #A0ADBD;
        font-size: 11px;
        line-height: 1.35;
        margin-bottom: 12px;
    }

    .fi-section {
        color: #F4F7FB;
        font-size: 13px;
        font-weight: 650;
        margin-top: 14px;
        margin-bottom: 9px;
    }

    [data-testid="stMetric"] {
        background: #1C2430;
        border: 1px solid #303D4D;
        border-radius: 12px;
        padding: 8px 12px;
    }

    [data-testid="stMetricLabel"] p {
        color: #A0ADBD !important;
        font-size: 11px !important;
    }

    [data-testid="stMetricValue"] {
        color: #F4F7FB !important;
        font-size: clamp(17px, 1.8vw, 23px) !important;
        font-weight: 700 !important;
    }

    [data-testid="stPlotlyChart"] {
        border: 1px solid #303D4D;
        border-radius: 12px;
        overflow: hidden;
    }

    [data-testid="stExpander"] {
        border: 1px solid #303D4D;
        border-radius: 10px;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    @media (max-width: 700px) {
        .fi-title {
            font-size: 22px;
        }

        .fi-section {
            font-size: 13px;
        }

        [data-testid="stMetricValue"] {
            font-size: 18px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DATA
# ============================================================

@st.cache_data(show_spinner="Loading PaySim dataset...")
def load_data():
    files = sorted(DATA_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            "PaySim CSV not found in data/"
        )

    return pd.read_csv(
        files[0],
        usecols=["step", "type", "amount", "isFraud"],
        dtype={
            "step": "int16",
            "type": "category",
            "amount": "float64",
            "isFraud": "int8",
        },
    )


df = load_data()


# ============================================================
# CHART HELPER
# ============================================================

def chart_style(fig, height):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=12, r=12, t=12, b=18),
        paper_bgcolor=C["surface"],
        plot_bgcolor=C["surface"],
        font=dict(color=C["muted"], size=10),
        showlegend=False,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            automargin=True,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=C["border"],
            zeroline=False,
            automargin=True,
        ),
    )
    return fig


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="fi-eyebrow">
        PAYSIM / FINANCIAL CRIME INTELLIGENCE
    </div>
    <div class="fi-title">
        Fraud Overview
    </div>
    <div class="fi-subtitle">
        Historical Transaction Analytics · Fraud Pattern Intelligence
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FILTERS
# ============================================================

with st.expander("⚙ Analytics Filters", expanded=False):
    filter_left, filter_right = st.columns(2)

    with filter_left:
        selected_types = st.multiselect(
            "Transaction types",
            options=list(df["type"].cat.categories),
            default=list(df["type"].cat.categories),
        )

    with filter_right:
        hour_range = st.slider(
            "Simulation hour",
            min_value=int(df["step"].min()),
            max_value=int(df["step"].max()),
            value=(
                int(df["step"].min()),
                int(df["step"].max()),
            ),
        )


filtered = df.loc[
    df["type"].isin(selected_types)
    & df["step"].between(*hour_range)
]


# ============================================================
# KPI
# ============================================================

total = len(filtered)
fraud = int(filtered["isFraud"].sum())
rate = fraud / total * 100 if total else 0.0

fraud_amount = filtered.loc[
    filtered["isFraud"].eq(1),
    "amount",
].sum()

k1, k2, k3, k4 = st.columns(4, gap="small")

k1.metric("Total Transactions", f"{total / 1e6:.2f}M")
k2.metric("Fraud Cases", f"{fraud:,}")
k3.metric("Fraud Rate", f"{rate:.3f}%")
k4.metric("Fraud Amount", f"{fraud_amount / 1e9:.2f}B")


# ============================================================
# TRANSACTION ACTIVITY
# ============================================================

st.markdown(
    '<div class="fi-section">Transaction Activity</div>',
    unsafe_allow_html=True,
)

activity_metric = st.radio(
    "Activity metric",
    ["Transactions", "Fraud Cases"],
    horizontal=True,
    label_visibility="collapsed",
)

activity = (
    filtered.groupby("step", observed=True)
    .agg(
        transactions=("isFraud", "size"),
        fraud_cases=("isFraud", "sum"),
    )
    .reset_index()
)

metric_col = (
    "transactions"
    if activity_metric == "Transactions"
    else "fraud_cases"
)

fig_activity = go.Figure()

fig_activity.add_trace(
    go.Scatter(
        x=activity["step"],
        y=activity[metric_col],
        mode="lines",
        line=dict(
            color=C["mint"],
            width=2,
        ),
        fill="tozeroy",
        fillcolor="rgba(84,214,193,0.10)",
        hovertemplate=(
            "Hour %{x}<br>"
            "Count %{y:,}<extra></extra>"
        ),
    )
)

st.plotly_chart(
    chart_style(fig_activity, height=200),
    use_container_width=True,
    config={"displayModeBar": False},
)


# ============================================================
# FRAUD BY TYPE
# ============================================================

by_type = (
    filtered.groupby("type", observed=True)
    .agg(
        total=("isFraud", "size"),
        fraud=("isFraud", "sum"),
    )
    .reset_index()
)

by_type["fraud_rate"] = (
    by_type["fraud"]
    .div(by_type["total"].replace(0, float("nan")))
    .fillna(0)
    * 100
)

left, right = st.columns(2, gap="small")

with left:
    st.markdown(
        '<div class="fi-section">Fraud Cases by Type</div>',
        unsafe_allow_html=True,
    )

    fig_cases = go.Figure(
        go.Bar(
            x=by_type["type"].astype(str),
            y=by_type["fraud"],
            marker_color=C["coral"],
            hovertemplate=(
                "%{x}<br>"
                "%{y:,} fraud cases<extra></extra>"
            ),
        )
    )

    st.plotly_chart(
        chart_style(fig_cases, height=180),
        use_container_width=True,
        config={"displayModeBar": False},
    )

with right:
    st.markdown(
        '<div class="fi-section">Fraud Rate by Type</div>',
        unsafe_allow_html=True,
    )

    fig_rate = go.Figure(
        go.Bar(
            x=by_type["type"].astype(str),
            y=by_type["fraud_rate"],
            marker_color=C["amber"],
            hovertemplate=(
                "%{x}<br>"
                "%{y:.3f}% fraud rate<extra></extra>"
            ),
        )
    )

    st.plotly_chart(
        chart_style(fig_rate, height=180),
        use_container_width=True,
        config={"displayModeBar": False},
    )

