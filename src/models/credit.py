from __future__ import annotations
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


def build_credit_pipeline(numeric_features, categorical_features, params=None):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    base = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.08,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "random_state": 42,
        "n_jobs": 4,
        "verbosity": 0,
    }
    if params:
        base.update(params)
    model = XGBClassifier(**base)
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def balanced_sample_weight(y):
    values, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    weights = {v: total / (len(values) * c) for v, c in zip(values, counts)}
    return np.asarray([weights[v] for v in y], dtype=float)
