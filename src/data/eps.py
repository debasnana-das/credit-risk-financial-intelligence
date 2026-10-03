from __future__ import annotations
import pandas as pd


def load_eps_data(path):
    df = pd.read_csv(path)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype(int)
    return df.sort_values(["Bank name", "Year"]).reset_index(drop=True)


def make_forecasting_table(df, feature_cols):
    work = df.sort_values(["Bank name", "Year"]).copy()
    work["Next-Year EPS (Rs.)"] = work.groupby("Bank name")["Basic EPS (Rs.)"].shift(-1)
    work["Next-Year"] = work.groupby("Bank name")["Year"].shift(-1)
    work = work.dropna(subset=["Next-Year EPS (Rs.)", "Next-Year"]).copy()
    work["Next-Year"] = work["Next-Year"].astype(int)
    work["Current EPS (Rs.)"] = work["Basic EPS (Rs.)"]
    features = feature_cols + ["Current EPS (Rs.)"]
    return work, features
