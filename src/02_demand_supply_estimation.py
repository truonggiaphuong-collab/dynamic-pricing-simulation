"""
Step 4: Demand and Supply Estimation

Estimates demand and supply variables from aggregated taxi data.
Uses proxies when real platform data is unavailable.

Usage:
    python src/02_demand_supply_estimation.py
"""

import pandas as pd
import numpy as np
from pathlib import Path


def estimate_demand_supply(df: pd.DataFrame) -> pd.DataFrame:
    """
    Refine demand and supply estimates from aggregated data.
    
    - Demand: pickup count (ride requests)
    - Supply: dropoff count from previous hour (drivers becoming available)
             With lag, supply[t] = drivers who finished trips in t-1
    """
    df = df.copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Demand: use pickup count directly (already in 'demand' from preprocessing)
    df['demand'] = df['demand'].fillna(0).clip(lower=0)
    
    # Supply: lag dropoffs by 1 hour (drivers finishing trips become available)
    # If 'driver_supply' is dropoff count, shift it: supply at t = dropoffs at t-1
    # Supply at t = dropoffs at t-1 (drivers becoming available). For first row, use current supply.
df['driver_supply'] = df['driver_supply'].shift(1).fillna(df['driver_supply'])
    
    # Ensure no zero supply (avoid division by zero in pricing)
    df['driver_supply'] = df['driver_supply'].clip(lower=1)
    
    # Optional: smooth with rolling mean to reduce noise
    df['demand_smooth'] = df['demand'].rolling(3, center=True, min_periods=1).mean()
    df['supply_smooth'] = df['driver_supply'].rolling(3, center=True, min_periods=1).mean()
    
    return df


def add_demand_supply_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Add demand/supply ratio - key input for surge pricing."""
    df = df.copy()
    df['demand_supply_ratio'] = df['demand'] / df['driver_supply']
    df['demand_supply_ratio_smooth'] = df['demand_smooth'] / df['supply_smooth']
    return df


def main():
    """Run demand/supply estimation pipeline."""
    data_dir = Path(__file__).parent.parent / 'data'
    input_path = data_dir / 'processed' / 'hourly_aggregated.csv'
    
    if not input_path.exists():
        print("Run 01_data_preprocessing.py first to create hourly_aggregated.csv")
        return
    
    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df = estimate_demand_supply(df)
    df = add_demand_supply_ratio(df)
    
    output_path = data_dir / 'processed' / 'demand_supply_estimated.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Saved to {output_path}")
    print(df[['timestamp', 'demand', 'driver_supply', 'demand_supply_ratio']].head(10).to_string())
    
    return df


if __name__ == '__main__':
    main()
