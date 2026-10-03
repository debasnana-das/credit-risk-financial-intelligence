# Credit tier model card

## Objective
Classify applicants into the four approval/priority tiers contained in `Approved_Flag`.

## Important semantic note
Public descriptions of this case-study dataset commonly interpret P1 as the lowest-risk/easiest-credit tier and P4 as the highest-risk/least-suitable tier. In this repository that interpretation is treated as a configurable case-study convention, **not** as an official bank policy. See the referenced 2025 paper: https://doi.org/10.1051/itmconf/20257001012

## Data
- 51,336 merged applicants
- 42 deployable features
- 5 categorical variables
- `-99999` is treated as missing
- Variables with >30% missingness in the training data are removed
- `PROSPECTID` is excluded from modeling
- `Credit_Score` is excluded from the deployable artifact because it is absent from the supplied unseen schema

## Validation
- 80/20 stratified holdout
- Model selection uses only training data
- Macro F1 and balanced accuracy are emphasized because P2 dominates the target distribution

## Final holdout snapshot
- Accuracy: 72.17%
- Balanced accuracy: 75.03%
- Macro F1: 0.687
- Weighted F1: 0.739

## Limitations
`Approved_Flag` is not an observed future default event. The model should therefore be described as **credit-tier/approval classification**, not as a calibrated probability-of-default model.
