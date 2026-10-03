# EPS forecasting model card

## Objective
Forecast one-year-ahead bank EPS from current-year financial indicators.

## Data
- 12 banks
- 2019–2023
- 60 bank-year observations
- 14 current-year financial indicators plus current EPS

## Validation
- The forecasting table is built with a one-year target shift within bank.
- Target year 2023 is the final untouched holdout.
- Earlier target years are used for expanding-window model selection.
- Current EPS carry-forward is the benchmark.

## Final holdout snapshot
Selected model: Random Forest regression.

- ML MAE: 9.13
- ML RMSE: 13.24
- ML R²: 0.611
- Current-EPS baseline MAE: 6.73
- Current-EPS baseline RMSE: 10.84
- Current-EPS baseline R²: 0.739

## Interpretation
The model does not beat the current-EPS baseline on the final holdout. That is a material finding, not something to hide. The dataset is too small to justify strong claims about future bank EPS forecasting.

## Safe positioning
Use the ML result as an experimental scenario forecast and demonstrate benchmark-first evaluation. Do not present it as investment advice or as evidence of reliable market forecasting.
