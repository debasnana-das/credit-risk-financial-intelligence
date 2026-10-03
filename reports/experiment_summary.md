# Experiment summary

## Credit tier classification

The deployable classifier uses 42 features shared with the supplied unseen-applicant schema. `Credit_Score` is excluded from the deployment artifact because it is not available in the unseen schema.

- Merged applicants: 51,336
- Train: 41,068
- Final holdout: 10,268
- Accuracy: 72.17%
- Balanced accuracy: 75.03%
- Macro F1: 0.687
- Weighted F1: 0.739

### Class-level F1

- P1: 0.767
- P2: 0.803
- P3: 0.446
- P4: 0.730

P3 is the hardest class in the supplied dataset, which is consistent with the overlapping credit-profile characteristics documented in public analyses of the same case study.

## EPS forecasting

The original same-year estimation setup was replaced with a one-year-ahead target shift.

- Banks: 12
- Years: 2019–2023
- Forecast target: next-year EPS
- Final holdout target year: 2023
- Selected model: Random Forest

### Final holdout

| Metric | ML | Current EPS baseline |
|---|---:|---:|
| MAE | 9.13 | 6.73 |
| RMSE | 13.24 | 10.84 |
| R² | 0.611 | 0.739 |

### Interpretation

The baseline is stronger on the untouched final holdout. The application therefore exposes both values instead of presenting the ML model as successful by default. This is a useful example of benchmark-first model evaluation on a small dataset.
