from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_SAMPLE = ROOT / "data" / "sample"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
METRICS = REPORTS / "metrics"

CREDIT_TARGET = "Approved_Flag"
CREDIT_ID = "PROSPECTID"
MISSING_SENTINEL = -99999
MISSING_THRESHOLD = 0.30

RISK_ORDER = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}
RISK_LABELS = {
    "P1": "Lower-risk tier",
    "P2": "Standard tier",
    "P3": "Higher-attention tier",
    "P4": "Highest-attention tier",
}

EPS_TARGET = "Basic EPS (Rs.)"
EPS_BANK = "Bank name"
EPS_YEAR = "Year"
EPS_CURRENT = "Current EPS (Rs.)"
EPS_NEXT = "Next-Year EPS (Rs.)"
EPS_FEATURES = [
    "ROCE (%)",
    "CASA (%)",
    "Net Profit Margin (%)",
    "Operating Profit Margin (%)",
    "Return on Assets (%)",
    "Return on Equity / Networth (%)",
    "Net Interest Margin (X)",
    "Cost to Income (%)",
    "Interest Income/Total Assets (%)",
    "Non-Interest Income/Total Assets (%)",
    "Operating Profit/Total Assets (%)",
    "Operating Expenses/Total Assets (%)",
    "Interest Expenses/Total Assets (%)",
    "Face_value",
]
