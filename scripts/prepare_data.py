from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "case_study1.xlsx": "credit_case_study1.csv",
    "case_study2.xlsx": "credit_case_study2.csv",
    "EPS_Dataset.xlsx": "eps_dataset.csv",
    "Unseen_Dataset.xlsx": "credit_unseen.csv",
    "Features_Target_Description.xlsx": "feature_dictionary.csv",
}

for source, dest in FILES.items():
    path = RAW / source
    if not path.exists():
        print(f"SKIP: {source} not found in {RAW}")
        continue
    df = pd.read_excel(path)
    df.to_csv(OUT / dest, index=False)
    print(f"Prepared {source} -> {dest} ({len(df):,} rows)")
