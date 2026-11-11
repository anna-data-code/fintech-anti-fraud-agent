"""PaySim | Risk Investigation — compact historical risk assessment."""
from pathlib import Path
import html
import pickle
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
APP_DIR = ROOT / 'app'
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from components.theme import apply_theme

st.set_page_config(page_title='Risk Investigation | PaySim', page_icon='🔎', layout='wide', initial_sidebar_state='expanded')
apply_theme()

FEATURES = [
    'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest',
    'newbalanceDest', 'errorBalanceOrig', 'errorBalanceDest', 'type_TRANSFER',
]
DATA_COLUMNS = [
    'step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig',
    'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFraud',
]

@st.cache_resource

def load_model():
    for path in (ROOT / 'xgboost_fraud_model.pkl', ROOT / 'models' / 'xgboost_fraud_model.pkl'):
        if path.is_file():
            with path.open('rb') as f:
                model = pickle.load(f)  # Load only your own trusted local model.
            if not hasattr(model, 'predict_proba'):
                raise TypeError('Model does not support predict_proba().')
            actual = list(getattr(model, 'feature_names_in_', FEATURES))
            if actual != FEATURES:
                raise ValueError(f'Model feature mismatch: {actual}')
            return model
    raise FileNotFoundError('xgboost_fraud_model.pkl not found in root or models/.')

@st.cache_data(show_spinner='Loading historical PaySim sample…')
def load_transactions():
    paths = sorted((ROOT / 'data').glob('*.csv')) or sorted((ROOT / 'data' / 'raw').glob('*.csv'))
    if not paths:
        raise FileNotFoundError('PaySim CSV not found in data/ or data/raw/.')
    return pd.read_csv(paths[0], usecols=DATA_COLUMNS, nrows=200_000)

def build_features(row):
    amount = float(row['amount'])
    old_org, new_org = float(row['oldbalanceOrg']), float(row['newbalanceOrig'])
    old_dest, new_dest = float(row['oldbalanceDest']), float(row['newbalanceDest'])
    values = {
        'step': int(row['step']), 'amount': amount,
        'oldbalanceOrg': old_org, 'newbalanceOrig': new_org,
        'oldbalanceDest': old_dest, 'newbalanceDest': new_dest,
        'errorBalanceOrig': old_org - amount - new_org,
        'errorBalanceDest': old_dest + amount - new_dest,
        'type_TRANSFER': int(row['type'] == 'TRANSFER'),
    }
    return pd.DataFrame([values], columns=FEATURES)

def safe(value):
    return html.escape(str(value), quote=True)

def risk_band(score):
    if score >= .8:
        return 'HIGH RISK', '#F17C94'
    if score >= .5:
        return 'ELEVATED', '#E9BE74'
    return 'LOWER SCORE', '#54D4BE'


st.markdown('''
<style>
  .block-container {
      padding-top: 1.15rem !important;
      padding-bottom: 1rem !important;
      max-width: 1380px;
  }

  div[data-testid="stVerticalBlock"] {
      gap: .72rem;
  }

  div[data-testid="stHorizontalBlock"] {
      gap: .9rem;
  }

  div[data-testid="stSelectbox"] label {
      font-size: .83rem !important;
      color: #A9B8C8 !important;
  }

  .fi-eyebrow {
      font-size: .70rem;
      letter-spacing: .15em;
      font-weight: 750;
      color: #54D4BE;
      margin: 0 0 5px;
  }

  .fi-heading {
      font-size: 1.8rem;
      line-height: 1.15;
      font-weight: 760;
      color: #F1F6FA;
      margin: 0;
  }

  .fi-sub {
      font-size: .85rem;
      color: #A9B8C8;
      margin: 6px 0 16px;
  }

  .fi-label {
      font-size: .82rem;
      font-weight: 750;
      color: #D5E1EC;
      margin: 13px 0 9px;
      line-height: 1.4;
  }

  .fi-card {
      background: linear-gradient(140deg,#1C2938,#202E40);
      border: 1px solid #334459;
      border-radius: 14px;
      padding: 19px 22px;
      min-height: 190px;
  }

  .fi-muted {
      font-size: .82rem;
      color: #A9B8C8;
  }

  .fi-big {
      font-size: 2.6rem;
      font-weight: 800;
      letter-spacing: -.045em;
      line-height: 1.18;
      margin: 12px 0;
  }

  .fi-track {
      height: 10px;
      background: #3A4A5B;
      border-radius: 100px;
      overflow: hidden;
  }

  .fi-status {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin: 12px 0 0;
      font-size: .78rem;
  }

  .fi-pair {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 12px 0;
      border-bottom: 1px solid #334459;
      font-size: .88rem;
  }

  .fi-pair:last-child {
      border-bottom: 0;
  }

  .fi-pair span {
      color: #A9B8C8;
  }

  .fi-pair strong {
      color: #F1F6FA;
      font-weight: 700;
  }

  .fi-profile {
      background: #1C2938;
      border: 1px solid #334459;
      border-radius: 14px;
      padding: 20px 22px;
  }

  .fi-profile-grid {
      display: grid;
      grid-template-columns: repeat(3,minmax(0,1fr));
      gap: 22px 28px;
  }

  .fi-field-name {
      font-size: .79rem;
      color: #A9B8C8;
      margin-bottom: 5px;
  }

  .fi-field-value {
      font-size: .98rem;
      color: #F1F6FA;
      font-weight: 650;
      overflow-wrap: anywhere;
  }

  @media (max-width: 850px) {
      .fi-profile-grid {
          grid-template-columns: repeat(2,minmax(0,1fr));
      }

      .fi-card {
          padding: 15px;
      }
  }
</style>
''', unsafe_allow_html=True)


try:
    model, df = load_model(), load_transactions()
except Exception as exc:
    st.error(f'Initialization failed: {exc}')
    st.stop()

st.markdown('''<div class="fi-eyebrow">PAYSIM / RISK INTELLIGENCE</div>
<div class="fi-heading">Risk Investigation</div>
<div class="fi-sub">Transaction-Level Risk Assessment · XGBoost Fraud Detection</div>''', unsafe_allow_html=True)

st.markdown('<div class="fi-label">SELECT TRANSACTION</div>', unsafe_allow_html=True)
sel1, sel2 = st.columns([1, 2.5], gap='small')
with sel1:
    mode = st.selectbox('Category', ['Fraud-labelled', 'Non-fraud-labelled', 'All transactions'])
if mode == 'Fraud-labelled':
    candidates = df[df['isFraud'] == 1]
elif mode == 'Non-fraud-labelled':
    candidates = df[df['isFraud'] == 0]
else:
    candidates = df
if candidates.empty:
    st.warning('No matching transactions in the loaded sample.')
    st.stop()
with sel2:
    selected_index = st.selectbox(
        'Transaction', candidates.index[:500].tolist(),
        format_func=lambda i: f"TX-{i:07d}  ·  {df.at[i, 'type']}  ·  {df.at[i, 'amount']:,.2f}",
    )

row = df.loc[selected_index]
X = build_features(row)
try:
    score = float(model.predict_proba(X)[0, 1])
except Exception as exc:
    st.error(f'Model prediction failed: {exc}')
    st.stop()

predicted = int(score >= .5)
actual = int(row['isFraud'])
label, color = risk_band(score)
pct = max(0., min(100., score * 100))
st.session_state['selected_transaction_index'] = int(selected_index)
st.session_state['selected_transaction_source'] = 'paysim_first_200k'

left, right = st.columns([1, 1], gap='small')
with left:
    st.markdown('<div class="fi-label">MODEL RISK SCORE</div>', unsafe_allow_html=True)
    st.markdown(f'''
    <div class="fi-card">
      <div class="fi-muted">XGBoost risk score · uncalibrated</div>
      <div class="fi-big" style="color:{color}">{pct:.2f}%</div>
      <div class="fi-track"><div style="width:{pct:.3f}%;height:100%;background:{color};border-radius:100px"></div></div>
      <div class="fi-status"><span style="color:{color};font-weight:750">{safe(label)}</span><span style="color:#A9B8C8">Illustrative threshold</span></div>
    </div>''', unsafe_allow_html=True)
with right:
    st.markdown('<div class="fi-label">MODEL ASSESSMENT</div>', unsafe_allow_html=True)
    match = 'Correct classification' if predicted == actual else 'Mismatch'
    match_color = '#54D4BE' if predicted == actual else '#F17C94'
    st.markdown(f'''
    <div class="fi-card">
      <div class="fi-pair"><span>Predicted class</span><strong>{'Fraud' if predicted else 'Non-fraud'}</strong></div>
      <div class="fi-pair"><span>Dataset label</span><strong>{'Fraud' if actual else 'Non-fraud'}</strong></div>
      <div class="fi-pair"><span>Comparison</span><strong style="color:{match_color}">{match}</strong></div>
    </div>''', unsafe_allow_html=True)

st.markdown('<div class="fi-label">TRANSACTION PROFILE</div>', unsafe_allow_html=True)
fields = [
    ('Transaction ID', f'TX-{selected_index:07d}'),
    ('Transaction type', row['type']),
    ('Amount (dataset units)', f"{row['amount']:,.2f}"),
    ('Simulation hour', int(row['step'])),
    ('Sender', row['nameOrig']),
    ('Receiver', row['nameDest']),
]
profile_html = ''.join(
    f'<div><div class="fi-field-name">{safe(k)}</div><div class="fi-field-value">{safe(v)}</div></div>'
    for k, v in fields
)
st.markdown(f'<div class="fi-profile"><div class="fi-profile-grid">{profile_html}</div></div>', unsafe_allow_html=True)

with st.expander('Balance details & model features', expanded=False):
    a, b = st.columns(2)
    with a:
        st.write(f"**Sender before:** {row['oldbalanceOrg']:,.2f}")
        st.write(f"**Sender after:** {row['newbalanceOrig']:,.2f}")
    with b:
        st.write(f"**Receiver before:** {row['oldbalanceDest']:,.2f}")
        st.write(f"**Receiver after:** {row['newbalanceDest']:,.2f}")
    st.dataframe(X.T.rename(columns={0: 'Feature value'}), width='stretch')

explain_page = ROOT / 'app' / 'pages' / '2_Explainable_AI.py'
if explain_page.is_file():
    if st.button('Explain this prediction with SHAP →', type='primary'):
        st.switch_page('pages/2_Explainable_AI.py')
else:
    st.caption('SHAP explanations will be available on the upcoming Explainable AI page.')

