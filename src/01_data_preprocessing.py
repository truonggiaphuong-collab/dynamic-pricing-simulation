"""
Step 3: Data Preprocessing for Dynamic Pricing Simulation

This script loads raw NYC taxi (or similar) data and transforms it into
the structure needed for demand/supply estimation and pricing simulation.

Usage:
    python src/01_data_preprocessing.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load raw taxi trip data from CSV.
    
    For NYC Yellow Taxi, columns include:
    - tpep_pickup_datetime, tpep_dropoff_datetime
    - PULocationID, DOLocationID
    - trip_distance, fare_amount, total_amount
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, nrows=100_000)  # Use nrows for testing; remove for full data
    return df


def preprocess_taxi_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare taxi data for aggregation.
    """
    # Parse datetime columns
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')
    
    # Drop rows with invalid timestamps
    df = df.dropna(subset=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])
    
    # Filter invalid trips
    df = df[df['trip_distance'] > 0]
    df = df[df['fare_amount'] > 0]
    df = df[df['total_amount'] > 0]
    
    # Remove outliers (optional)
    df = df[df['trip_distance'] < 100]  # Trips > 100 miles likely errors
    df = df[df['fare_amount'] < 500]    # Fares > $500 likely errors
    
    return df


def aggregate_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate trip-level data to hourly buckets for demand/supply analysis.
    
    Creates:
    - demand: count of pickups per hour (ride requests)
    - driver_supply: count of dropoffs per hour (proxy for completed trips / driver availability)
    - trip_distance: mean distance per hour
    - base_price: mean fare_amount (or use fixed base)
    - total_amount: mean total charged
    """
    df = df.copy()
    df['hour'] = df['tpep_pickup_datetime'].dt.floor('H')
    
    # Demand = pickups in that hour (customers requesting rides)
    demand_agg = df.groupby('hour').agg(
        pickup_count=('tpep_pickup_datetime', 'count')
    ).reset_index()
    
    # For supply: use dropoffs in previous hour as proxy for "drivers becoming available"
    # Or use pickup count as both (simplified: supply = trips completed)
    df['dropoff_hour'] = df['tpep_dropoff_datetime'].dt.floor('H')
    supply_agg = df.groupby('dropoff_hour').agg(
        dropoff_count=('tpep_dropoff_datetime', 'count')
    ).reset_index().rename(columns={'dropoff_hour': 'hour'})
    
    # Trip metrics
    trip_agg = df.groupby('hour').agg(
        trip_distance=('trip_distance', 'mean'),
        base_price=('fare_amount', 'mean'),
        total_amount=('total_amount', 'mean')
    ).reset_index()
    
    # Merge
    agg_df = demand_agg.merge(supply_agg, on='hour', how='outer').fillna(0)
    agg_df = agg_df.merge(trip_agg, on='hour', how='left')
    
    # Rename for clarity
    agg_df = agg_df.rename(columns={
        'hour': 'timestamp',
        'pickup_count': 'demand',
        'dropoff_count': 'driver_supply'
    })
    
    return agg_df


def create_sample_data() -> pd.DataFrame:
    """
    Create synthetic sample data when real data is not available.
    Useful for testing the pipeline without downloading large files.
    """
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', '2024-01-31', freq='H')
    
    # Simulate demand: higher during rush hours (7-9, 17-19)
    demand = np.random.poisson(50, len(dates))
    for i, d in enumerate(dates):
        if 7 <= d.hour <= 9 or 17 <= d.hour <= 19:
            demand[i] = int(demand[i] * 2.5 + np.random.poisson(30))
        if 0 <= d.hour <= 9:
            demand[i] = max(10, int(demand[i] * 0.6))
    
    # Supply: slightly lagged, fewer drivers at night
    supply = np.random.poisson(45, len(dates))
    for i, d in enumerate(dates):
        if 7 <= d.hour <= 9 or 17 <= d.hour <= 19:
            supply[i] = int(supply[i] * 1.2)
        if 0 <= d.hour <= 5:
            supply[i] = max(5, int(supply[i] * 0.5))
    
    # Ensure supply is never zero to avoid division issues
    supply = np.maximum(supply, 1)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'demand': demand,
        'driver_supply': supply,
        'trip_distance': np.random.uniform(2, 10, len(dates)).round(2),
        'base_price': np.random.uniform(8, 25, len(dates)).round(2),
        'total_amount': np.random.uniform(10, 35, len(dates)).round(2)
    })
    
    return df


def main():
    """Run preprocessing pipeline."""
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    
    raw_path = data_dir / 'raw' / 'yellow_tripdata_2024-01.csv'
    
    if raw_path.exists():
        df = load_raw_data(str(raw_path))
        df = preprocess_taxi_data(df)
        agg_df = aggregate_to_hourly(df)
    else:
        print("Raw data not found. Creating synthetic sample data...")
        agg_df = create_sample_data()
    
    output_path = data_dir / 'processed' / 'hourly_aggregated.csv'
    output_path.parent.mkdir(exist_ok=True)
    agg_df.to_csv(output_path, index=False)
    
    print(f"\nProcessed data saved to {output_path}")
    print(f"Shape: {agg_df.shape}")
    print(agg_df.head(10).to_string())
    
    return agg_df


if __name__ == '__main__':
    main()
