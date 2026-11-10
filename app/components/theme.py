
import streamlit as st

COLORS = {
    "bg": "#11151D",
    "surface": "#1C2430",
    "border": "#303D4D",
    "text": "#F4F7FB",
    "muted": "#A0ADBD",
    "mint": "#54D6C1",
    "coral": "#F17888",
    "amber": "#F0BE6D",
}


def apply_theme():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #11151D;
            color: #F4F7FB;
        }

        .block-container {
            max-width: 1450px;
            padding-top: 1rem;
            padding-bottom: 1rem;
        }

        header[data-testid="stHeader"] {
            background: transparent;
            height: 0;
        }

        #MainMenu, footer {
            visibility: hidden;
        }

        section[data-testid="stSidebar"] {
            background: #171E29;
            border-right: 1px solid #303D4D;
        }

        section[data-testid="stSidebar"] a {
            border-radius: 9px;
        }

        section[data-testid="stSidebar"] a[aria-current="page"] {
            background: #263B3C;
        }

        .fi-eyebrow {
            color: #54D6C1;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 1.5px;
            margin-bottom: 5px;
        }

        .fi-title {
            color: #F4F7FB;
            font-size: 26px;
            font-weight: 750;
            line-height: 1.2;
            margin-bottom: 5px;
        }

        .fi-subtitle {
            color: #A0ADBD;
            font-size: 12px;
            margin-bottom: 12px;
        }

        .fi-section {
            color: #F4F7FB;
            font-size: 14px;
            font-weight: 650;
            margin: 8px 0;
        }

        [data-testid="stMetric"] {
            background: #1C2430;
            border: 1px solid #303D4D;
            border-radius: 12px;
            padding: 12px;
        }

        [data-testid="stMetricLabel"] p {
            color: #A0ADBD !important;
            font-size: 11px !important;
        }

        [data-testid="stMetricValue"] {
            color: #F4F7FB !important;
            font-size: 21px !important;
            font-weight: 700 !important;
        }

        [data-testid="stPlotlyChart"] {
            background: #1C2430;
            border: 1px solid #303D4D;
            border-radius: 12px;
            overflow: hidden;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid #303D4D;
            border-radius: 12px;
        }

        .fi-note {
            color: #A0ADBD;
            font-size: 11px;
            line-height: 1.5;
            padding-top: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title, subtitle):
    st.markdown(
        f"""
        <div class="fi-eyebrow">
            PAYSIM / FINANCIAL CRIME INTELLIGENCE
        </div>
        <div class="fi-title">{title}</div>
        <div class="fi-subtitle">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title):
    st.markdown(
        f'<div class="fi-section">{title}</div>',
        unsafe_allow_html=True,
    )
