# Portfolio positioning

## Resume-ready description

**Banking Risk & Financial Intelligence — Integrated Credit Tier Assessment & Financial Forecasting**

Built a deployable banking analytics platform combining customer credit-tier classification with one-year-ahead bank EPS forecasting. Developed a leakage-aware XGBoost classification pipeline over 51K+ merged credit records, exposed class probabilities, created time-aware EPS forecasting with baseline comparison, and packaged both models behind Streamlit and FastAPI interfaces.

## Interview talking points

1. Why two separate models? The inputs operate at different levels: customer-level credit behaviour and bank-level financial performance.
2. Why not call the credit output PD? The supplied target is an approval/priority tier, not an observed future default event.
3. Why exclude Credit_Score from the deployable artifact? It is absent from the supplied unseen-applicant schema, so deployment-schema parity matters more than squeezing out a benchmark metric.
4. Why keep the baseline for EPS? Because the model must prove incremental value; on this small dataset the current-EPS baseline wins.
5. Why Streamlit first? It creates a visible interactive product directly from the GitHub repository; FastAPI is included as the engineering/API layer.
