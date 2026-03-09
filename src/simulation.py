"""
Simulation engine for dynamic pricing over time.

Runs pricing simulation for a full day or multiple days.
Tracks: demand, supply, price multiplier, revenue, rides served.
Computes summary statistics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from .data_processing import generate_synthetic_data, aggregate_to_hourly, prepare_ml_features
from .demand_model import evaluate_models, get_best_model
from .pricing_model import apply_pricing, PricingConfig


def run_simulation(
    n_days: int = 7,
    use_predicted_demand: bool = True,
    config: Optional[PricingConfig] = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """
    Run full dynamic pricing simulation.

    Args:
        n_days: Number of days to simulate.
        use_predicted_demand: If True, use ML-predicted demand for pricing.
        config: Pricing config. Default: min=1, max=3.
        seed: Random seed for data generation.

    Returns:
        (results_df, summary_dict) with simulation results and statistics.
    """
    config = config or PricingConfig(min_multiplier=1.0, max_multiplier=3.0)

    # Generate data
    end_date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=n_days)
    df_raw = generate_synthetic_data(
        start_date="2024-01-01",
        end_date=end_date.strftime("%Y-%m-%d"),
        seed=seed,
    )
    df = aggregate_to_hourly(df_raw)

    # Train demand models and get predictions
    X, y = prepare_ml_features(df)
    models, metrics = evaluate_models(X, y)
    best_name = get_best_model(metrics, criterion="RMSE")
    best_model = models[best_name]
    df["demand_predicted"] = best_model.predict(X)

    # Apply pricing (use predicted or actual demand)
    demand_col = "demand_predicted" if use_predicted_demand else "demand"
    df = apply_pricing(df, config, demand_col=demand_col)

    # Compute rides served (min of demand and supply) and revenue
    df["rides_served"] = np.minimum(df["demand"], df["driver_supply"])
    df["revenue"] = df["rides_served"] * df["price"]

    # Summary statistics
    summary = {
        "total_revenue": df["revenue"].sum(),
        "avg_price": df["price"].mean(),
        "avg_multiplier": df["surge_multiplier"].mean(),
        "total_demand": df["demand"].sum(),
        "total_rides_served": df["rides_served"].sum(),
        "demand_served_ratio": df["rides_served"].sum() / df["demand"].sum() if df["demand"].sum() > 0 else 0,
        "best_demand_model": best_name,
        "model_metrics": metrics,
    }

    return df, summary


def save_results(df: pd.DataFrame, summary: dict, output_dir: Path) -> None:
    """Save simulation results and summary to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "simulation_results.csv", index=False)
    with open(output_dir / "simulation_summary.txt", "w") as f:
        for k, v in summary.items():
            if isinstance(v, dict) and k == "model_metrics":
                f.write(f"{k}:\n")
                for model_name, m in v.items():
                    f.write(f"  {model_name}: MAE={m['MAE']:.2f}, RMSE={m['RMSE']:.2f}, R2={m['R2']:.3f}\n")
            elif isinstance(v, dict):
                f.write(f"{k}:\n")
                for k2, v2 in v.items():
                    f.write(f"  {k2}: {v2}\n")
            else:
                f.write(f"{k}: {v}\n")
