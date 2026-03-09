"""
Dynamic pricing model for ride-hailing platforms.

Implements surge pricing based on demand/supply ratio:
    price_multiplier = predicted_demand / supply
    multiplier = clip(multiplier, min=1, max=3)
    price = base_price * multiplier
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class PricingConfig:
    """Configuration for the dynamic pricing model."""

    min_multiplier: float = 1.0
    max_multiplier: float = 3.0
    baseline_ratio: float = 1.0


def compute_multiplier(
    demand: float,
    supply: float,
    config: PricingConfig,
) -> float:
    """
    Compute surge multiplier from demand/supply ratio.

    Args:
        demand: Predicted or actual demand (ride requests).
        supply: Available driver supply.
        config: Pricing constraints.

    Returns:
        Surge multiplier in [min_multiplier, max_multiplier].
    """
    if supply <= 0:
        return config.max_multiplier
    ratio = demand / supply
    multiplier = ratio / config.baseline_ratio
    return float(np.clip(multiplier, config.min_multiplier, config.max_multiplier))


def apply_pricing(
    df: pd.DataFrame,
    config: PricingConfig,
    demand_col: str = "demand",
    supply_col: str = "driver_supply",
    base_price_col: str = "base_price",
) -> pd.DataFrame:
    """
    Apply dynamic pricing to a DataFrame.

    Adds columns: surge_multiplier, price.

    Args:
        df: DataFrame with demand, supply, base_price columns.
        config: Pricing configuration.
        demand_col: Column name for demand.
        supply_col: Column name for supply.
        base_price_col: Column name for base price.

    Returns:
        DataFrame with surge_multiplier and price columns added.
    """
    df = df.copy()
    supply = df[supply_col].replace(0, np.nan).fillna(1)
    ratio = df[demand_col] / supply
    df["surge_multiplier"] = (ratio / config.baseline_ratio).clip(
        config.min_multiplier,
        config.max_multiplier,
    )
    df["price"] = df[base_price_col] * df["surge_multiplier"]
    return df
