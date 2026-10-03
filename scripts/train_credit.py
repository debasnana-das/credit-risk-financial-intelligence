from __future__ import annotations
import json
from pathlib import Path
import sys
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import ARTIFACTS, METRICS, FIGURES, MISSING_THRESHOLD
from src.data.credit import load_credit_data
from src.models.credit import balanced_sample_weight, build_credit_pipeline
from src.utils.io import save_json, save_joblib

CASE1 = ROOT / "data" / "processed" / "credit_case_study1.csv"
CASE2 = ROOT / "data" / "processed" / "credit_case_study2.csv"
UNSEEN = ROOT / "data" / "processed" / "credit_unseen.csv"


def evaluate(name, model, X_test, y_test, labels):
    pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro"),
        "weighted_f1": f1_score(y_test, pred, average="weighted"),
        "macro_precision": precision_score(y_test, pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_test, pred, average="macro", zero_division=0),
        "classification_report": classification_report(y_test, pred, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=labels).tolist(),
    }
    return metrics


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True); METRICS.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    df = load_credit_data(CASE1, CASE2)
    # Keep rows with the target; missing predictor values are imputed inside the pipeline.
    target = "Approved_Flag"
    y_raw = df[target].astype(str)
    feature_source = df.drop(columns=[target])
    feature_source = feature_source.drop(columns=["PROSPECTID"], errors="ignore")

    # The supplied unseen set has no Credit_Score, so the deployed model deliberately uses the common schema.
    unseen = pd.read_csv(UNSEEN)
    unseen_common = [c for c in unseen.columns if c in feature_source.columns]
    feature_source = feature_source[unseen_common].replace(-99999, np.nan)
    unseen = unseen[unseen_common].replace(-99999, np.nan)

    cat = feature_source.select_dtypes(include=["object"]).columns.tolist()
    num = [c for c in feature_source.columns if c not in cat]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    X_train, X_test, y_train, y_test = train_test_split(
        feature_source, y, test_size=0.20, random_state=42, stratify=y
    )

    # Learn the missingness filter from training data only to keep the holdout fully untouched.
    missing_rates = X_train.isna().mean()
    drop_missing = [c for c, r in missing_rates.items() if r > MISSING_THRESHOLD]
    X_train = X_train.drop(columns=drop_missing)
    X_test = X_test.drop(columns=drop_missing)
    unseen = unseen.drop(columns=drop_missing, errors="ignore")
    num = [c for c in num if c not in drop_missing]
    cat = [c for c in cat if c not in drop_missing]
    sample_weight = balanced_sample_weight(y_train)

    # Baseline: majority class.
    dummy = Pipeline([
        ("preprocess", ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), num),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), cat),
        ])),
        ("model", DummyClassifier(strategy="most_frequent"))
    ])
    dummy.fit(X_train, y_train)
    baseline = evaluate("majority_baseline", dummy, X_test, y_test, list(range(len(le.classes_))))

    base = build_credit_pipeline(num, cat)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    param_dist = {
        "model__n_estimators": [150, 200, 250],
        "model__max_depth": [3, 4, 5],
        "model__learning_rate": [0.04, 0.06, 0.08, 0.12],
        "model__subsample": [0.75, 0.85, 1.0],
        "model__colsample_bytree": [0.7, 0.85, 1.0],
        "model__min_child_weight": [1, 3, 6],
        "model__reg_alpha": [0.0, 0.1, 0.5, 1.0],
        "model__reg_lambda": [1.0, 2.0, 5.0],
    }
    search = RandomizedSearchCV(
        base, param_distributions=param_dist, n_iter=4, scoring="f1_macro",
        cv=2, random_state=42, n_jobs=1, verbose=1, refit=True
    )
    search.fit(X_train, y_train, model__sample_weight=sample_weight)
    model = search.best_estimator_
    credit_metrics = evaluate("xgboost", model, X_test, y_test, list(range(len(le.classes_))))

    # Fit metadata on full training data only after model selection, then save the fitted pipeline.
    save_joblib(model, ARTIFACTS / "credit_tier_model.joblib")
    save_json({
        "classes": le.classes_.tolist(),
        "feature_columns": feature_source.columns.tolist(),
        "numeric_columns": num,
        "categorical_columns": cat,
        "dropped_high_missing_columns": drop_missing,
        "unseen_feature_columns": unseen_common,
        "best_params": search.best_params_,
        "thresholds": {"missing_rate_drop": MISSING_THRESHOLD},
    }, ARTIFACTS / "credit_metadata.json")
    save_json({"majority_baseline": baseline, "xgboost": credit_metrics}, METRICS / "credit_metrics.json")

    cm = np.array(credit_metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm)
    ax.set_xticks(range(len(le.classes_))); ax.set_yticks(range(len(le.classes_)))
    ax.set_xticklabels(le.classes_); ax.set_yticklabels(le.classes_)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Credit Tier Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]): ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(FIGURES / "credit_confusion_matrix.png", dpi=160); plt.close(fig)

    # Feature importance after one-hot encoding.
    prep = model.named_steps["preprocess"]
    xgb = model.named_steps["model"]
    names = prep.get_feature_names_out()
    importances = xgb.feature_importances_
    top = pd.Series(importances, index=names).sort_values(ascending=False).head(15).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    top.plot(kind="barh", ax=ax)
    ax.set_title("Top Credit Model Features"); ax.set_xlabel("Importance")
    fig.tight_layout(); fig.savefig(FIGURES / "credit_feature_importance.png", dpi=160); plt.close(fig)

    # Score the supplied unseen set; this is a reproducible end-to-end artifact.
    unseen_pred_idx = model.predict(unseen)
    unseen_prob = model.predict_proba(unseen)
    scored = unseen.copy()
    scored["Predicted_Tier"] = le.inverse_transform(unseen_pred_idx)
    for i, cls in enumerate(le.classes_): scored[f"Prob_{cls}"] = unseen_prob[:, i]
    scored.to_csv(ROOT / "reports" / "unseen_credit_scores.csv", index=False)

    summary = {
        "rows_merged": int(len(df)),
        "features_before_filtering": int(df.shape[1] - 1),
        "features_used": int(feature_source.shape[1]),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "target_distribution": y_raw.value_counts().to_dict(),
        "best_params": search.best_params_,
        "test_metrics": {k: v for k, v in credit_metrics.items() if k not in ["classification_report", "confusion_matrix"]},
    }
    save_json(summary, METRICS / "credit_summary.json")
    print(json.dumps(summary, indent=2, default=str))

if __name__ == "__main__":
    main()
