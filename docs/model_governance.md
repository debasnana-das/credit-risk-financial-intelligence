# Model governance

This repository intentionally separates predictive modeling from business policy.

## Credit model
The trained classifier predicts the supplied P1–P4 approval/priority label. It is not a default model because the dataset does not provide a future default event. A future production implementation should use a clearly defined outcome horizon, e.g. default within 12 months, and then calibrate PD.

## EPS model
The financial dataset contains only 12 banks across five years. The model is therefore treated as an experimental forecasting component. The current-EPS carry-forward baseline is always shown so users can see whether ML adds value.

## Human oversight
The combined context layer is a transparent rule engine. It never silently converts model output into a loan approval, rejection or investment recommendation.

## Monitoring to add in production
- feature drift and missingness drift
- target/performance drift
- calibration monitoring for probabilities
- subgroup performance and fairness review
- model/version/audit logging
- threshold approval and change control
