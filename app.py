"""Run with: python -m streamlit run app.py"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from risklab.service import CreditRequest, RiskService

st.set_page_config(page_title="Credit risk review", layout="wide")
st.title("Credit risk review")
st.caption("Historical UCI data · research demo · amounts in New Taiwan dollars")
report = Path("reports/metrics.json")
if report.exists():
    metrics = json.loads(report.read_text())
    selected = metrics["test"][metrics["selected_model"]]
    a, b, c = st.columns(3)
    a.metric("Test ROC-AUC", f"{selected['roc_auc']:.3f}")
    b.metric("Test average precision", f"{selected['average_precision']:.3f}")
    c.metric("Test Brier score", f"{selected['brier']:.3f}")
try:
    service = st.cache_resource(RiskService)()
except (OSError, ValueError):
    st.info("Train the model first: python -m risklab.train")
    st.stop()
st.subheader("Explore a synthetic customer profile")
st.write(
    "Enter six months in order from most recent to oldest. Statuses are recorded dataset codes (-2 to 9); positive values indicate months delayed. Keep nonpositive codes as supplied."
)
with st.form("profile"):
    limit = st.number_input("Credit limit", min_value=1.0, max_value=10000000.0, value=100000.0)
    table = pd.DataFrame(
        {
            "Month": ["Sep", "Aug", "Jul", "Jun", "May", "Apr"],
            "Repayment status": [0] * 6,
            "Bill amount": [30000.0] * 6,
            "Payment amount": [4000.0] * 6,
        }
    )
    table = st.data_editor(table, hide_index=True, disabled=["Month"])
    submitted = st.form_submit_button("Score profile")
if submitted:
    try:
        request = CreditRequest(
            limit_balance=limit,
            repayment_status=[int(x) for x in table["Repayment status"]],
            bill_amounts=table["Bill amount"].tolist(),
            payment_amounts=table["Payment amount"].tolist(),
        )
        result = service.predict(request)
        st.metric(
            "Estimated next-month default probability",
            f"{result['default_probability']:.1%}",
        )
        st.write(
            f"Review policy: **{result['action']}** · threshold {result['review_threshold']:.2f}"
        )
        st.caption(result["notice"])
    except ValueError as exc:
        st.error(str(exc))
if report.exists():
    with st.expander("Validation findings and subgroup audit"):
        st.image("reports/validation.svg")
        st.dataframe(pd.read_csv("reports/subgroup_audit.csv"), hide_index=True)
        st.caption(
            "Differences across groups require investigation. Excluding demographics does not establish fairness."
        )
