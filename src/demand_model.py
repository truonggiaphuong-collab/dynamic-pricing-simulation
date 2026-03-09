"""
Demand prediction models for ride-hailing dynamic pricing.

Implements:
- Baseline: Linear Regression
- Tree-based: Random Forest
- Advanced: XGBoost / Gradient Boosting

Evaluates using MAE, RMSE, R².
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    HAS_XGBOOST = False


def get_feature_columns() -> list[str]:
    """Return standard feature columns for demand prediction."""
    return ["hour_sin", "hour_cos", "day_of_week", "trip_distance"]


def train_linear_regression(X: pd.DataFrame, y: pd.Series) -> tuple[Any, dict]:
    """
    Train baseline Linear Regression model.

    Returns:
        (model, metrics_dict) with MAE, RMSE, R².
    """
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    metrics = _compute_metrics(y, y_pred)
    return model, metrics


def train_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 10,
    random_state: int = 42,
) -> tuple[Any, dict]:
    """
    Train Random Forest regressor.

    Returns:
        (model, metrics_dict) with MAE, RMSE, R².
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(X, y)
    y_pred = model.predict(X)
    metrics = _compute_metrics(y, y_pred)
    return model, metrics


def train_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42,
) -> tuple[Any, dict]:
    """
    Train XGBoost (or Gradient Boosting fallback) model.

    Returns:
        (model, metrics_dict) with MAE, RMSE, R².
    """
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
    model.fit(X, y)
    y_pred = model.predict(X)
    metrics = _compute_metrics(y, y_pred)
    return model, metrics


def _compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, R²."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[dict[str, Any], dict[str, dict]]:
    """
    Train and evaluate all three demand prediction models.

    Returns:
        (models_dict, metrics_dict) where models_dict maps model name to fitted model,
        and metrics_dict maps model name to {MAE, RMSE, R2}.
    """
    models = {}
    metrics = {}

    lr_model, lr_metrics = train_linear_regression(X, y)
    models["Linear Regression"] = lr_model
    metrics["Linear Regression"] = lr_metrics

    rf_model, rf_metrics = train_random_forest(X, y)
    models["Random Forest"] = rf_model
    metrics["Random Forest"] = rf_metrics

    xgb_model, xgb_metrics = train_xgboost(X, y)
    models["XGBoost"] = xgb_model
    metrics["XGBoost"] = xgb_metrics

    return models, metrics


def get_best_model(metrics: dict[str, dict], criterion: str = "RMSE") -> str:
    """
    Return the name of the best model based on the given criterion.

    For RMSE and MAE, lower is better. For R2, higher is better.
    """
    if criterion.upper() == "R2":
        return max(metrics.keys(), key=lambda k: metrics[k][criterion])
    return min(metrics.keys(), key=lambda k: metrics[k][criterion])
