"""
Simulation engine for dynamic pricing over time.

Runs pricing simulation with train/test validation.
Includes counterfactual: surge vs no-surge comparison.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from .data_processing import (
    load_and_preprocess,
    generate_synthetic_data,
    aggregate_to_hourly,
    prepare_ml_features,
    train_test_split_temporal,
)
from .demand_model import evaluate_models_with_split, get_best_model
from .pricing_model import apply_pricing, PricingConfig


def run_simulation(
    n_days: int = 7,
    use_predicted_demand: bool = True,
    config: Optional[PricingConfig] = None,
    seed: int = 42,
    test_ratio: float = 0.2,
    use_nyc_data: bool = True,
    data_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Run full dynamic pricing simulation with validation and counterfactual.

    Returns:
        (results_df, summary_dict)
    """
    config = config or PricingConfig(min_multiplier=1.0, max_multiplier=3.0)
    data_dir = data_dir or Path(__file__).parent.parent / "data"

    # Load data (try NYC, fallback to synthetic)
    df, data_source = load_and_preprocess(
        use_nyc_download=use_nyc_data,
        data_dir=data_dir,
        n_days_synthetic=n_days,
    )

    # Ensure enough rows; if NYC gave few, pad with synthetic
    if len(df) < 100:
        df_raw = generate_synthetic_data(
            start_date="2024-01-01",
            end_date="2024-01-08",
            seed=seed,
        )
        df = aggregate_to_hourly(df_raw)
        data_source = "synthetic"

    # Train/test split (time-based)
    df_train, df_test = train_test_split_temporal(df, test_ratio=test_ratio)
    X_train, y_train = prepare_ml_features(df_train)
    X_test, y_test = prepare_ml_features(df_test)

    # Train models and evaluate on test set
    models, train_metrics, test_metrics = evaluate_models_with_split(
        X_train, y_train, X_test, y_test
    )
    best_name = get_best_model(test_metrics, criterion="RMSE")
    best_model = models[best_name]

    # Predict on full dataset for simulation
    X_full, _ = prepare_ml_features(df)
    df["demand_predicted"] = best_model.predict(X_full)

    # Apply dynamic pricing
    demand_col = "demand_predicted" if use_predicted_demand else "demand"
    df = apply_pricing(df, config, demand_col=demand_col)
    df["rides_served"] = np.minimum(df["demand"], df["driver_supply"])
    df["revenue"] = df["rides_served"] * df["price"]

    # Counterfactual: no surge (multiplier = 1)
    df["price_no_surge"] = df["base_price"]
    df["revenue_no_surge"] = df["rides_served"] * df["price_no_surge"]

    total_revenue_surge = df["revenue"].sum()
    total_revenue_no_surge = df["revenue_no_surge"].sum()
    revenue_lift = (total_revenue_surge - total_revenue_no_surge) / total_revenue_no_surge if total_revenue_no_surge > 0 else 0

    summary = {
        "total_revenue": total_revenue_surge,
        "total_revenue_no_surge": total_revenue_no_surge,
        "revenue_lift_pct": revenue_lift * 100,
        "avg_price": df["price"].mean(),
        "avg_multiplier": df["surge_multiplier"].mean(),
        "total_demand": df["demand"].sum(),
        "total_rides_served": df["rides_served"].sum(),
        "demand_served_ratio": df["rides_served"].sum() / df["demand"].sum() if df["demand"].sum() > 0 else 0,
        "best_demand_model": best_name,
        "data_source": data_source,
        "model_metrics_train": train_metrics,
        "model_metrics_test": test_metrics,
    }

    return df, summary


def save_results(df: pd.DataFrame, summary: dict, output_dir: Path) -> None:
    """Save simulation results and summary to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "simulation_results.csv", index=False)
    with open(output_dir / "simulation_summary.txt", "w") as f:
        for k, v in summary.items():
            if isinstance(v, dict) and "model_metrics" in str(k):
                f.write(f"{k}:\n")
                for model_name, m in v.items():
                    f.write(f"  {model_name}: MAE={m['MAE']:.2f}, RMSE={m['RMSE']:.2f}, R2={m['R2']:.3f}\n")
            elif isinstance(v, dict):
                f.write(f"{k}:\n")
                for k2, v2 in v.items():
                    f.write(f"  {k2}: {v2}\n")
            else:
                f.write(f"{k}: {v}\n")
