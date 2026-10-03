from __future__ import annotations
import numpy as np
import pandas as pd


def load_credit_data(case1_path, case2_path):
    a = pd.read_csv(case1_path)
    b = pd.read_csv(case2_path)
    df = a.merge(b, on="PROSPECTID", how="inner", validate="one_to_one")
    df = df.replace(-99999, np.nan)
    return df


def prepare_credit(df: pd.DataFrame, include_credit_score: bool = False):
    work = df.copy()
    drop_cols = [c for c in work.columns if c not in ["Approved_Flag"] and work[c].isna().mean() > 0.30]
    work = work.drop(columns=drop_cols)
    cols = [c for c in work.columns if c not in ["Approved_Flag", "PROSPECTID"]]
    if not include_credit_score:
        cols = [c for c in cols if c != "Credit_Score"]
    X = work[cols]
    y = work["Approved_Flag"]
    cat = X.select_dtypes(include=["object"]).columns.tolist()
    num = [c for c in X.columns if c not in cat]
    return X, y, num, cat, drop_cols
