"""Regression model definitions (Linear Regression, Random Forest, XGBoost)."""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


def get_regression_models(seed: int = 42) -> dict:
    """Return the regression model catalogue.

    Preprocessing (feature scaling) is embedded inside a scikit-learn
    Pipeline so it is fitted on training data only and travels with the
    serialized model — no leakage, no forgotten scaler at inference time.
    """
    return {
        "Linear Regression": Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=4,
            random_state=seed,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        ),
    }
