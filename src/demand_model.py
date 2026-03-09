"""
Demand prediction models for ride-hailing dynamic pricing.

Implements: Linear Regression, Random Forest, XGBoost.
Uses train/test split for proper out-of-sample evaluation.
"""

import numpy as np
import pandas as pd
from typing import Any, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    HAS_XGBOOST = False


def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """Train baseline Linear Regression."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 10,
    random_state: int = 42,
) -> Any:
    """Train Random Forest regressor."""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42,
) -> Any:
    """Train XGBoost or Gradient Boosting fallback."""
    if HAS_XGBOOST:
        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )
    model.fit(X_train, y_train)
    return model


def _compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, R²."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def evaluate_models_with_split(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[dict[str, Any], dict[str, dict], dict[str, dict]]:
    """
    Train on train set, evaluate on both train and test.

    Returns:
        (models_dict, train_metrics_dict, test_metrics_dict)
    """
    models = {}
    train_metrics = {}
    test_metrics = {}

    for name, train_fn in [
        ("Linear Regression", lambda: train_linear_regression(X_train, y_train)),
        ("Random Forest", lambda: train_random_forest(X_train, y_train)),
        ("XGBoost", lambda: train_xgboost(X_train, y_train)),
    ]:
        model = train_fn()
        models[name] = model
        train_metrics[name] = _compute_metrics(y_train, model.predict(X_train))
        test_metrics[name] = _compute_metrics(y_test, model.predict(X_test))

    return models, train_metrics, test_metrics


def get_best_model(metrics: dict[str, dict], criterion: str = "RMSE") -> str:
    """Return name of best model by criterion (lower RMSE/MAE, higher R2)."""
    if criterion.upper() == "R2":
        return max(metrics.keys(), key=lambda k: metrics[k][criterion])
    return min(metrics.keys(), key=lambda k: metrics[k][criterion])
