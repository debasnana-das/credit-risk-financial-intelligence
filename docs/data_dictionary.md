# Data dictionary

The original variable dictionary is included in the source material as `Features_Target_Description.xlsx` and is converted locally to `data/processed/feature_dictionary.csv`.

### Credit data
The two credit workbooks contain customer/account history and CIBIL/external-credit variables, joined on `PROSPECTID`.

Key groups include:

- Account/tradeline history: active/closed accounts, account opening/closure recency, missed payments, secured/unsecured exposure.
- Delinquency: delinquency counts, DPD counts, sub-standard/doubtful/loss indicators.
- Enquiries: total, credit-card, personal-loan and recency-based enquiry counts.
- Customer profile: age, gender, marital status, education, income, employment tenure.
- Product behaviour: credit-card/personal-loan flags, product enquiries and utilization metrics.
- Target: `Approved_Flag` with four classes P1–P4.

### Financial data
`EPS_Dataset.xlsx` contains annual bank-level indicators such as ROCE, CASA, profit margins, ROA, ROE, net interest margin, cost-to-income, asset ratios, interest expenses, face value and Basic EPS.
