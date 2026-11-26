"""Classification model definitions (Logistic Regression, Random Forest, XGBoost)."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def get_classification_models(seed: int = 42) -> dict:
    """Return the UP/DOWN classification model catalogue.

    Logistic Regression needs scaled features, so it is wrapped in a Pipeline
    with a StandardScaler fitted on training data only. Random Forest and
    XGBoost are scale-invariant. All models expose ``predict_proba`` so the
    dashboard can show probability scores.
    """
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=2000, random_state=seed),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=4,
            random_state=seed,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        ),
    }
