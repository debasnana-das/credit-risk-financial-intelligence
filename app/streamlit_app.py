from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
SAMPLE = ROOT / "data" / "sample"

st.set_page_config(page_title="Banking Risk & Financial Intelligence", page_icon="🏦", layout="wide")

@st.cache_resource
def load_models():
    import joblib
    credit = joblib.load(ART / "credit_tier_model.joblib")
    eps = joblib.load(ART / "eps_forecast_model.joblib")
    credit_meta = json.loads((ART / "credit_metadata.json").read_text())
    eps_meta = json.loads((ART / "eps_metadata.json").read_text())
    return credit, eps, credit_meta, eps_meta

@st.cache_data
def load_reference():
    return pd.read_csv(ART / "bank_latest_reference.csv")

credit_model, eps_model, credit_meta, eps_meta = load_models()
reference = load_reference()
financial_features = [f for f in eps_meta["features"] if f != "Current EPS (Rs.)"]

st.markdown("# Banking Risk & Financial Intelligence")
st.markdown("**Customer credit-tier assessment · next-year EPS forecasting · decision context**")
st.caption("Portfolio project built from the supplied credit-modeling and bank-financial datasets. Outputs are analytical decision support, not automated lending advice.")

with st.sidebar:
    st.markdown("### System")
    st.write("**Credit model:** XGBoost multiclass")
    st.write(f"**Financial model:** {eps_meta['selected_model'].replace('_', ' ').title()}")
    st.write("**Credit classes:** P1–P4")
    st.write("**Forecast horizon:** one year")
    st.divider()
    st.info("The financial model is intentionally benchmarked against a current-EPS baseline because the source dataset contains only 12 banks × 5 years.")

tab1, tab2, tab3, tab4 = st.tabs(["Customer Credit", "Bank Financial Outlook", "Combined Context", "Model & Data Cards"])

with tab1:
    st.subheader("Customer credit-tier assessment")
    st.write("Score a single applicant from a CSV row using the same feature schema as the trained model, or use the included demo profile.")

    demo = pd.read_csv(SAMPLE / "credit_applicants_demo.csv")
    mode = st.radio("Input mode", ["Demo applicant", "Upload CSV"], horizontal=True)

    if mode == "Demo applicant":
        idx = st.number_input("Demo applicant row", min_value=1, max_value=len(demo), value=1, step=1) - 1
        row = demo.iloc[[idx]].copy()
        st.dataframe(row.T.rename(columns={row.index[0]: "value"}), use_container_width=True)
    else:
        uploaded = st.file_uploader("Upload applicant CSV", type=["csv"])
        if uploaded is None:
            st.warning("Upload a CSV containing the 42 model features.")
            row = None
        else:
            row = pd.read_csv(uploaded)
            missing = [c for c in credit_meta["feature_columns"] if c not in row.columns]
            if missing:
                st.error(f"Missing required columns ({len(missing)}): {missing}")
                row = None
            else:
                if len(row) != 1:
                    st.warning("Please upload a CSV containing exactly one applicant row. Batch scoring is not enabled in this dashboard.")
                    row = None
                else:
                    row = row[credit_meta["feature_columns"]]

    if row is not None and st.button("Assess credit tier", type="primary"):
        row = row.replace(-99999, pd.NA)
        pred = credit_model.predict(row)[0]
        probs = credit_model.predict_proba(row)[0]
        classes = credit_meta["classes"]
        predicted_tier = str(classes[pred])
        st.session_state["last_credit_tier"] = predicted_tier

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted tier", predicted_tier)
        c2.metric("Tier attention", {"P1":"Lower", "P2":"Standard", "P3":"Higher", "P4":"Highest"}.get(predicted_tier, "Unknown"))
        c3.metric("Highest class probability", f"{probs.max()*100:.1f}%")
        prob_df = pd.DataFrame({"Tier": classes, "Probability": probs}).sort_values("Probability", ascending=False)
        st.bar_chart(prob_df.set_index("Tier"))
        st.dataframe(prob_df.style.format({"Probability": "{:.1%}"}), use_container_width=True)
        if predicted_tier in {"P3", "P4"}:
            st.warning("Higher-attention tier. Treat the output as a screening signal and apply the institution's documented lending policy and human review.")
        else:
            st.success("Standard/lower-attention tier signal. The model is not a substitute for the institution's lending policy.")

with tab2:
    st.subheader("Bank financial outlook")
    st.write("Forecast the next year's EPS from the current year's financial indicators. Selecting a supplied 2023 row creates a 2024 scenario forecast.")
    banks = reference["Bank name"].tolist()
    bank = st.selectbox("Bank", banks)
    current = reference.loc[reference["Bank name"] == bank].iloc[0]

    st.caption(f"Reference year: {int(current['Year'])}. Edit inputs to create a scenario.")
    values = {}
    grid = st.columns(2)
    for i, feature in enumerate(financial_features):
        with grid[i % 2]:
            default = float(current[feature])
            values[feature] = st.number_input(feature, value=default, format="%.4f", key=f"eps_{feature}")
    current_eps_input = st.number_input("Current EPS (Rs.)", value=float(current["Basic EPS (Rs.)"]), format="%.4f", key="eps_current")

    if st.button("Forecast next-year EPS", type="primary"):
        X = pd.DataFrame([{**values, "Current EPS (Rs.)": float(current_eps_input)}])
        forecast = float(eps_model.predict(X)[0])

        # Use the exact user-entered Current EPS for both growth and the carry-forward baseline.
        curr = float(current_eps_input)
        baseline = curr
        growth = ((forecast / curr) - 1) * 100 if abs(curr) > 1e-9 else None

        a, b, c = st.columns(3)
        a.metric("Current EPS", f"₹{curr:.2f}")
        b.metric("ML forecast", f"₹{forecast:.2f}")
        c.metric("Forecast growth", "N/A" if growth is None else f"{growth:+.1f}%")
        st.info(f"Benchmark baseline: carrying forward the entered current EPS would be ₹{baseline:.2f}. This baseline is included because the validation study found it stronger than the ML forecast on the untouched 2023 holdout.")
        if growth is not None and growth < -5:
            st.warning("Negative forecast scenario. Use this as a sensitivity signal, not a standalone investment or credit conclusion.")
        elif growth is not None and growth > 5:
            st.success("Positive forecast scenario. Verify assumptions and compare against the baseline before using the result.")
        else:
            st.info("Stable forecast scenario.")

with tab3:
    st.subheader("Combined scenario context")
    st.write("This layer connects the two models without pretending that bank-level EPS directly determines an individual customer's credit risk.")
    default_tier = st.session_state.get("last_credit_tier", "P2")
    tier_index = credit_meta["classes"].index(default_tier) if default_tier in credit_meta["classes"] else 1
    credit_tier = st.selectbox("Customer tier", credit_meta["classes"], index=tier_index, key="combined_credit_tier")
    bank = st.selectbox("Institution", banks, key="combined_bank")
    current = reference.loc[reference["Bank name"] == bank].iloc[0]
    X = pd.DataFrame([{**{f: float(current[f]) for f in financial_features}, "Current EPS (Rs.)": float(current["Basic EPS (Rs.)"])}])
    forecast = float(eps_model.predict(X)[0])
    curr = float(current["Basic EPS (Rs.)"])
    growth = ((forecast / curr) - 1) * 100 if abs(curr) > 1e-9 else 0.0

    if credit_tier in {"P3", "P4"}:
        posture = "Enhanced credit review"
    else:
        posture = "Standard credit review"
    if growth < -5:
        posture += " + closer portfolio monitoring"

    a,b,c = st.columns(3)
    a.metric("Customer tier", credit_tier)
    b.metric("EPS scenario", f"₹{forecast:.2f}")
    c.metric("Scenario posture", posture)
    st.warning("This is a communication/portfolio-context rule, not a trained underwriting policy. Financial outlook must not be used to override customer-level credit evidence.")

    matrix = pd.DataFrame([
        ["P1", "Standard review", "Enhanced monitoring if financial outlook weak"],
        ["P2", "Standard review", "Standard review + portfolio monitoring if outlook weak"],
        ["P3", "Enhanced review", "Enhanced review + portfolio monitoring"],
        ["P4", "Highest attention", "Highest attention + portfolio monitoring"],
    ], columns=["Customer tier", "Positive / stable outlook", "Negative outlook"])
    st.dataframe(matrix, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Model card")
    m1, m2 = st.columns(2)
    with m1:
        st.markdown("#### Credit tier model")
        credit_metrics = json.loads((ROOT / "reports/metrics/credit_metrics.json").read_text())["xgboost"]
        st.metric("Accuracy", f"{credit_metrics['accuracy']:.1%}")
        st.metric("Macro F1", f"{credit_metrics['macro_f1']:.3f}")
        st.metric("Balanced accuracy", f"{credit_metrics['balanced_accuracy']:.3f}")
        st.caption("20% stratified holdout; tuning performed only on training folds. Credit_Score is excluded from the deployable model because the supplied unseen schema does not contain it.")
    with m2:
        st.markdown("#### EPS forecast model")
        eps_metrics = json.loads((ROOT / "reports/metrics/eps_metrics.json").read_text())
        st.metric("2023 holdout MAE", f"{eps_metrics['holdout_2023_ml']['MAE']:.2f}")
        st.metric("2023 holdout RMSE", f"{eps_metrics['holdout_2023_ml']['RMSE']:.2f}")
        st.metric("Baseline MAE", f"{eps_metrics['holdout_2023_current_eps_baseline']['MAE']:.2f}")
        st.caption("Time-aware training uses earlier years; 2023 is untouched for final evaluation. The current-EPS baseline remains stronger on this small dataset.")
    st.divider()
    st.markdown("#### Why the limitations matter")
    st.write("The credit target is a four-level approval/priority label rather than an observed future default outcome. The EPS dataset contains only 12 banks over 5 years. The project therefore emphasizes leakage control, baseline comparison, deployment parity, and transparent limitations rather than overstating model performance.")
