from __future__ import annotations
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor


def candidate_models(n_features):
    numeric = list(range(n_features))
    return {
        "ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]),
        "gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(
                n_estimators=80, learning_rate=0.04, max_depth=2,
                min_samples_leaf=3, random_state=42
            )),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=250, max_depth=4, min_samples_leaf=2,
                max_features=0.8, random_state=42, n_jobs=4
            )),
        ]),
        "xgboost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBRegressor(
                objective="reg:squarederror", n_estimators=120,
                max_depth=2, learning_rate=0.04, min_child_weight=3,
                subsample=0.85, colsample_bytree=0.8, reg_alpha=1.0,
                reg_lambda=5.0, random_state=42, n_jobs=4
            )),
        ]),
    }
