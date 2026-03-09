"""
Step 5 & 6: Dynamic Pricing Model Implementation

Economics-based surge pricing:
    price_multiplier = demand / supply
    constrained to [min_multiplier, max_multiplier]

Final price = base_price * price_multiplier

Usage:
    python src/03_pricing_model.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass


@dataclass
class PricingConfig:
    """Configuration for the dynamic pricing model."""
    min_multiplier: float = 1.0
    max_multiplier: float = 3.0
    baseline_ratio: float = 1.0  # Ratio at which multiplier = 1


def compute_surge_multiplier(
    demand: float,
    supply: float,
    config: PricingConfig
) -> float:
    """
    Compute surge multiplier from demand/supply ratio.
    
    Formula: multiplier = demand / supply
    Constraints: min_multiplier <= multiplier <= max_multiplier
    
    When demand == supply, multiplier = 1 (no surge).
    When demand >> supply, multiplier approaches max_multiplier.
    """
    if supply <= 0:
        return config.max_multiplier  # Safety: max surge if no supply
    
    raw_ratio = demand / supply
    
    # Scale: at baseline_ratio, we want multiplier = 1
    # So: multiplier = raw_ratio / baseline_ratio
    multiplier = raw_ratio / config.baseline_ratio
    
    # Apply constraints
    multiplier = np.clip(multiplier, config.min_multiplier, config.max_multiplier)
    
    return float(multiplier)


def apply_pricing_model(
    df: pd.DataFrame,
    config: PricingConfig,
    demand_col: str = 'demand',
    supply_col: str = 'driver_supply',
    base_price_col: str = 'base_price'
) -> pd.DataFrame:
    """
    Apply dynamic pricing model to entire dataframe.
    
    Adds columns:
    - surge_multiplier
    - simulated_price (= base_price * surge_multiplier)
    """
    df = df.copy()
    
    multipliers = []
    for _, row in df.iterrows():
        mult = compute_surge_multiplier(
            row[demand_col],
            row[supply_col],
            config
        )
        multipliers.append(mult)
    
    df['surge_multiplier'] = multipliers
    df['simulated_price'] = df[base_price_col] * df['surge_multiplier']
    
    return df


def apply_pricing_vectorized(
    df: pd.DataFrame,
    config: PricingConfig,
    demand_col: str = 'demand',
    supply_col: str = 'driver_supply',
    base_price_col: str = 'base_price'
) -> pd.DataFrame:
    """
    Vectorized version - faster for large datasets.
    """
    df = df.copy()
    
    ratio = df[demand_col] / df[supply_col].replace(0, np.nan)
    ratio = ratio.fillna(1.0)  # No surge when supply is zero or missing
    
    df['surge_multiplier'] = (ratio / config.baseline_ratio).clip(
        config.min_multiplier,
        config.max_multiplier
    )
    df['simulated_price'] = df[base_price_col] * df['surge_multiplier']
    
    return df


def main():
    """Run pricing model on processed data."""
    data_dir = Path(__file__).parent.parent / 'data'
    input_path = data_dir / 'processed' / 'demand_supply_estimated.csv'
    
    if not input_path.exists():
        print("Run 01_data_preprocessing.py and 02_demand_supply_estimation.py first")
        return
    
    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    config = PricingConfig(min_multiplier=1.0, max_multiplier=3.0)
    df = apply_pricing_vectorized(df, config)
    
    output_path = data_dir / 'processed' / 'pricing_simulation.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Pricing simulation saved to {output_path}")
    print(f"\nSurge multiplier stats:\n{df['surge_multiplier'].describe()}")
    print(f"\nSample:\n{df[['timestamp', 'demand', 'driver_supply', 'surge_multiplier', 'simulated_price']].head(10).to_string()}")
    
    return df


if __name__ == '__main__':
    main()
