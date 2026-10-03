from __future__ import annotations
import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import ARTIFACTS, METRICS, FIGURES, EPS_FEATURES, EPS_TARGET
from src.data.eps import load_eps_data, make_forecasting_table
from src.models.eps import candidate_models
from src.utils.io import save_joblib, save_json

EPS_PATH = ROOT / "data" / "processed" / "eps_dataset.csv"


def expanding_splits(df):
    # Returns train/validation masks using target years. 2023 remains the final holdout.
    years = sorted(df["Next-Year"].unique())
    train_years = years
    for val_year in train_years[1:]:
        train = df["Next-Year"] < val_year
        val = df["Next-Year"] == val_year
        yield train, val, val_year


def score(y_true, pred):
    return {
        "MAE": mean_absolute_error(y_true, pred),
        "RMSE": mean_squared_error(y_true, pred) ** 0.5,
        "R2": r2_score(y_true, pred) if len(np.unique(y_true)) > 1 else float("nan"),
    }


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True); METRICS.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    raw = load_eps_data(EPS_PATH)
    table, features = make_forecasting_table(raw, EPS_FEATURES)
    # 2023 target remains untouched for final evaluation; model selection uses only earlier target years.
    trainval = table[table["Next-Year"] <= 2022].copy()
    final_test = table[table["Next-Year"] == 2023].copy()
    X_tv, y_tv = trainval[features], trainval["Next-Year EPS (Rs.)"]
    X_test, y_test = final_test[features], final_test["Next-Year EPS (Rs.)"]

    model_scores = {name: [] for name in candidate_models(len(features))}
    candidates = candidate_models(len(features))
    for train_mask, val_mask, year in expanding_splits(trainval):
        # Ensure chronological ordering within each split.
        X_tr = trainval.loc[train_mask, features]; y_tr = trainval.loc[train_mask, "Next-Year EPS (Rs.)"]
        X_va = trainval.loc[val_mask, features]; y_va = trainval.loc[val_mask, "Next-Year EPS (Rs.)"]
        for name, model in candidates.items():
            model.fit(X_tr, y_tr)
            pred = model.predict(X_va)
            s = score(y_va, pred); s["validation_year"] = int(year); model_scores[name].append(s)

    cv_summary = {}
    for name, rows in model_scores.items():
        cv_summary[name] = {
            "mean_MAE": float(np.mean([r["MAE"] for r in rows])),
            "mean_RMSE": float(np.mean([r["RMSE"] for r in rows])),
            "mean_R2": float(np.nanmean([r["R2"] for r in rows])),
            "folds": rows,
        }
    selected = min(cv_summary, key=lambda n: cv_summary[n]["mean_MAE"])

    final_model = candidates[selected]
    final_model.fit(X_tv, y_tv)
    pred = final_model.predict(X_test)
    ml_metrics = score(y_test, pred)
    baseline_pred = final_test["Current EPS (Rs.)"].to_numpy()
    baseline_metrics = score(y_test, baseline_pred)

    save_joblib(final_model, ARTIFACTS / "eps_forecast_model.joblib")
    reference = raw.sort_values(["Bank name", "Year"]).groupby("Bank name").tail(1).copy()
    reference = reference[["Bank name", "Year"] + EPS_FEATURES + [EPS_TARGET]]
    reference.to_csv(ROOT / "artifacts" / "bank_latest_reference.csv", index=False)
    metadata = {
        "features": features,
        "selected_model": selected,
        "historical_years": sorted(raw["Year"].unique().tolist()),
        "final_holdout_target_year": 2023,
        "forecast_horizon": "one year",
    }
    save_json(metadata, ARTIFACTS / "eps_metadata.json")
    save_json({"cross_validation": cv_summary, "holdout_2023_ml": ml_metrics, "holdout_2023_current_eps_baseline": baseline_metrics}, METRICS / "eps_metrics.json")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    plot_df = final_test[["Bank name"]].copy()
    plot_df["Actual"] = y_test.values
    plot_df["ML Forecast"] = pred
    plot_df["Baseline"] = baseline_pred
    plot_df = plot_df.sort_values("Actual")
    x = np.arange(len(plot_df))
    ax.plot(x, plot_df["Actual"], marker="o", label="Actual 2023 EPS")
    ax.plot(x, plot_df["ML Forecast"], marker="o", label="ML forecast")
    ax.plot(x, plot_df["Baseline"], marker="o", label="Current EPS baseline")
    ax.set_xticks(x); ax.set_xticklabels(plot_df["Bank name"], rotation=45, ha="right")
    ax.set_ylabel("EPS (Rs.)"); ax.set_title("2023 EPS Holdout: Actual vs Forecast")
    ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "eps_holdout_2023.png", dpi=160); plt.close(fig)

    out = final_test[["Bank name", "Next-Year"]].copy()
    out["Actual EPS"] = y_test.values
    out["ML Forecast EPS"] = pred
    out["Current EPS Baseline"] = baseline_pred
    out.to_csv(ROOT / "reports" / "eps_2023_holdout_predictions.csv", index=False)

    latest_2023 = raw[raw["Year"] == 2023][["Bank name"] + EPS_FEATURES + [EPS_TARGET]].copy()
    latest_X = latest_2023[EPS_FEATURES + [EPS_TARGET]].rename(columns={EPS_TARGET: "Current EPS (Rs.)"})
    latest_pred = final_model.predict(latest_X)
    forecast_2024 = latest_2023[["Bank name", EPS_TARGET]].copy()
    forecast_2024 = forecast_2024.rename(columns={EPS_TARGET: "Current EPS (Rs.)"})
    forecast_2024["Forecast EPS (2024) (Rs.)"] = latest_pred
    forecast_2024["Forecast Growth (%)"] = np.where(
        forecast_2024["Current EPS (Rs.)"].abs() > 1e-9,
        (forecast_2024["Forecast EPS (2024) (Rs.)"] / forecast_2024["Current EPS (Rs.)"] - 1) * 100,
        np.nan,
    )
    forecast_2024.to_csv(ROOT / "reports" / "forecast_2024_demo.csv", index=False)

    summary = {
        "rows": int(len(raw)),
        "banks": int(raw["Bank name"].nunique()),
        "training_target_years": sorted(trainval["Next-Year"].unique().tolist()),
        "holdout_year": 2023,
        "selected_model": selected,
        "holdout_ml": ml_metrics,
        "holdout_baseline": baseline_metrics,
    }
    save_json(summary, METRICS / "eps_summary.json")
    print(json.dumps(summary, indent=2, default=str))

if __name__ == "__main__":
    main()
