"""
Data processing pipeline for ride-hailing dynamic pricing simulation.

Generates and preprocesses data with columns:
- timestamp, hour, day_of_week, location_id
- trip_distance, demand, driver_supply, base_price

Supports both synthetic data generation and raw NYC taxi data loading.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


def generate_synthetic_data(
    start_date: str = "2024-01-01",
    end_date: str = "2024-01-31",
    n_locations: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic synthetic ride-hailing data for simulation.

    Simulates demand/supply patterns:
    - Higher demand during rush hours (7-9 AM, 5-7 PM)
    - Lower supply at night (12-5 AM)
    - Location-specific variation (e.g., airport vs downtown)

    Args:
        start_date: Start of simulation period.
        end_date: End of simulation period.
        n_locations: Number of geographic zones.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: timestamp, hour, day_of_week, location_id,
        trip_distance, demand, driver_supply, base_price.
    """
    np.random.seed(seed)
    dates = pd.date_range(start_date, end_date, freq="H")
    n_periods = len(dates)

    records = []
    for i, ts in enumerate(dates):
        for loc_id in range(n_locations):
            # Base demand: higher in rush hours
            base_demand = np.random.poisson(30)
            if 7 <= ts.hour <= 9 or 17 <= ts.hour <= 19:
                base_demand = int(base_demand * 2.2 + np.random.poisson(25))
            if 0 <= ts.hour <= 5:
                base_demand = max(5, int(base_demand * 0.4))

            # Location effect: airport (0) and downtown (1) have higher demand
            loc_factor = 1.2 if loc_id in (0, 1) else 0.9
            demand = max(1, int(base_demand * loc_factor * np.random.uniform(0.8, 1.2)))

            # Supply: fewer drivers at night
            base_supply = np.random.poisson(35)
            if 7 <= ts.hour <= 9 or 17 <= ts.hour <= 19:
                base_supply = int(base_supply * 1.3)
            if 0 <= ts.hour <= 5:
                base_supply = max(3, int(base_supply * 0.5))
            supply = max(1, int(base_supply * np.random.uniform(0.9, 1.1)))

            # Trip distance and base price (distance-based)
            trip_distance = np.random.uniform(2, 12)
            base_price = 2.5 + trip_distance * 1.5 + np.random.uniform(-1, 2)

            records.append({
                "timestamp": ts,
                "hour": ts.hour,
                "day_of_week": ts.dayofweek,
                "location_id": loc_id,
                "trip_distance": round(trip_distance, 2),
                "demand": demand,
                "driver_supply": supply,
                "base_price": round(max(5, base_price), 2),
            })

    df = pd.DataFrame(records)
    return df


def aggregate_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate trip-level data to hourly buckets (for location-agnostic analysis).

    Sums demand and supply across locations per hour.
    """
    df = df.copy()
    df["hour_bucket"] = df["timestamp"].dt.floor("H")

    agg = df.groupby("hour_bucket").agg(
        demand=("demand", "sum"),
        driver_supply=("driver_supply", "sum"),
        trip_distance=("trip_distance", "mean"),
        base_price=("base_price", "mean"),
    ).reset_index()
    agg = agg.rename(columns={"hour_bucket": "timestamp"})
    agg["hour"] = agg["timestamp"].dt.hour
    agg["day_of_week"] = agg["timestamp"].dt.dayofweek
    agg["driver_supply"] = agg["driver_supply"].clip(lower=1)
    return agg


def load_and_preprocess(raw_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load data from raw file or generate synthetic data.

    Args:
        raw_path: Path to raw NYC taxi CSV. If None, generates synthetic data.

    Returns:
        Preprocessed DataFrame ready for modeling.
    """
    if raw_path and raw_path.exists():
        df = pd.read_csv(raw_path, nrows=100_000)
        df = _preprocess_nyc_taxi(df)
        return aggregate_to_hourly(df)
    return _get_synthetic_aggregated()


def _preprocess_nyc_taxi(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess NYC TLC taxi trip data."""
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"], errors="coerce")
    df = df.dropna(subset=["tpep_pickup_datetime", "tpep_dropoff_datetime"])
    df = df[(df["trip_distance"] > 0) & (df["fare_amount"] > 0)]
    df = df[(df["trip_distance"] < 100) & (df["fare_amount"] < 500)]
    df["timestamp"] = df["tpep_pickup_datetime"]
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["location_id"] = (df["PULocationID"] % 5) if "PULocationID" in df.columns else 0
    df["demand"] = 1  # Each row = 1 trip
    df["driver_supply"] = 1
    df["trip_distance"] = df["trip_distance"]
    df["base_price"] = df["fare_amount"]
    return df[["timestamp", "hour", "day_of_week", "location_id", "trip_distance", "demand", "driver_supply", "base_price"]]


def _get_synthetic_aggregated() -> pd.DataFrame:
    """Generate synthetic data and aggregate to hourly."""
    df = generate_synthetic_data()
    return aggregate_to_hourly(df)


def prepare_ml_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for demand prediction models.

    Features: hour, day_of_week, trip_distance, (location_id if present)
    Target: demand

    Returns:
        (X, y) tuple for model training.
    """
    df = df.copy()
    feature_cols = ["hour", "day_of_week", "trip_distance"]
    if "location_id" in df.columns:
        feature_cols.append("location_id")
    X = df[[c for c in feature_cols if c in df.columns]].copy()
    X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24)
    X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24)
    if "hour" in X.columns:
        X = X.drop(columns=["hour"])
    y = df["demand"]
    return X, y
