import numpy as np
import pandas as pd
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

def calculate_custom_features(amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest):
    errorBalanceOrig = oldbalanceOrg - amount - newbalanceOrig
    errorBalanceDest = oldbalanceDest + amount - newbalanceDest
    
    return {
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        'errorBalanceOrig': errorBalanceOrig,
        'errorBalanceDest': errorBalanceDest
    }

def generate_interactive_shap_chart(top_shap):
    """
    Creates a modern, trendy minimalist Plotly horizontal bar chart.
    """
    features = [item['feature'] for item in top_shap]
    impacts = [item['shap_val'] for item in top_shap]
    # Trendy fintech palette: Deep slate blue for risk, muted soft blue for mitigation
    colors = ['#38bdf8' if x > 0 else '#64748b' for x in impacts]

    fig = go.Figure(go.Bar(
        x=impacts,
        y=features,
        orientation='h',
        marker=dict(color=colors, cornerradius=6),
        hoverinfo='x+y'
    ))

    fig.update_layout(
        title=dict(text="<b>SHAP Feature Attribution</b>", font=dict(size=13, color="#f8fafc")),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#94a3b8", family="Inter, sans-serif"),
        margin=dict(l=10, r=10, t=35, b=10),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.1)'),
        yaxis=dict(showgrid=False),
        height=210
    )
    return fig

def generate_llm_sar_narrative(transaction_data, risk_score, top_shap_features):
    narrative = f"""
    SUSPICIOUS ACTIVITY REPORT (SAR) NARRATIVE
    --------------------------------------------------
    Automated Flagging Reason: Risk Index ({risk_score*100:.1f}%)
    
    Incident Summary:
    Transaction involving account step {transaction_data.get('step', 'N/A')} with an amount of ${transaction_data['amount']:,.2f} 
    was flagged by the PaySim AI Engine.
    
    Key Primary Risk Drivers (SHAP Attribution):
    1. {top_shap_features[0]['feature']}: Value = {top_shap_features[0]['value']}, Impact = {top_shap_features[0]['shap_val']:.4f}
    2. {top_shap_features[1]['feature']}: Value = {top_shap_features[1]['value']}, Impact = {top_shap_features[1]['shap_val']:.4f}
    3. {top_shap_features[2]['feature']}: Value = {top_shap_features[2]['value']}, Impact = {top_shap_features[2]['shap_val']:.4f}
    
    Compliance Recommendation:
    Immediate automated hold placed. Routed to compliance team for review.
    --------------------------------------------------
    Report Generated automatically by PaySim XAI Engine.
    """
    return narrative.strip()

def create_pdf_report(narrative_text):
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    textobject = p.beginText()
    textobject.setTextOrigin(50, 750)
    textobject.setFont("Helvetica", 10)
    
    for line in narrative_text.split('\n'):
        textobject.textLine(line)
        
    p.drawText(textobject)
    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer