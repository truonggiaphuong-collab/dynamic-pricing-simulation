"""
Data processing pipeline for ride-hailing dynamic pricing simulation.

Generates and preprocesses data with columns:
- timestamp, hour, day_of_week, location_id
- trip_distance, demand, driver_supply, base_price

Supports NYC TLC taxi data (with download) and synthetic data fallback.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

# NYC TLC data URL (official cloudfront)
NYC_TAXI_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"


def download_nyc_taxi_sample(
    url: str = NYC_TAXI_URL,
    output_dir: Optional[Path] = None,
    max_rows: int = 150_000,
) -> Optional[pd.DataFrame]:
    """
    Download a sample of NYC TLC Yellow Taxi data.

    Falls back to None if download fails (e.g., no network).

    Args:
        url: URL to parquet file.
        output_dir: Optional directory to cache the file.
        max_rows: Maximum rows to load (for memory).

    Returns:
        DataFrame with trip records, or None if download fails.
    """
    try:
        import requests
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        from io import BytesIO
        df = pd.read_parquet(BytesIO(resp.content))
        df = df.head(max_rows)
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_dir / "yellow_tripdata_sample.parquet", index=False)
        return df
    except Exception:
        return None


def load_nyc_taxi(raw_path: Path) -> pd.DataFrame:
    """Load and preprocess NYC taxi data from local file (parquet or CSV)."""
    raw_path = Path(raw_path)
    if raw_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(raw_path)
    else:
        df = pd.read_csv(raw_path, nrows=150_000)
    return _preprocess_nyc_taxi(df)


def aggregate_nyc_taxi_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate NYC trip-level data to hourly demand/supply.

    - Demand: pickup count per hour
    - Supply: dropoff count from previous hour (drivers becoming available)
    """
    df = df.copy()
    df["pickup_hour"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce").dt.floor("h")
    df["dropoff_hour"] = pd.to_datetime(df["tpep_dropoff_datetime"], errors="coerce").dt.floor("h")

    demand_agg = df.groupby("pickup_hour").agg(
        demand=("tpep_pickup_datetime", "count"),
        trip_distance=("trip_distance", "mean"),
        base_price=("fare_amount", "mean"),
    ).reset_index().rename(columns={"pickup_hour": "timestamp"})

    supply_agg = df.groupby("dropoff_hour").agg(
        dropoff_count=("tpep_dropoff_datetime", "count"),
    ).reset_index().rename(columns={"dropoff_hour": "hour_bucket"})
    supply_agg["timestamp"] = supply_agg["hour_bucket"] + pd.Timedelta(hours=1)
    supply_agg = supply_agg[["timestamp", "dropoff_count"]].rename(columns={"dropoff_count": "driver_supply"})

    agg = demand_agg.merge(supply_agg, on="timestamp", how="left")
    agg["driver_supply"] = agg["driver_supply"].fillna(agg["demand"]).clip(lower=1)
    agg["hour"] = agg["timestamp"].dt.hour
    agg["day_of_week"] = agg["timestamp"].dt.dayofweek
    return agg


def generate_synthetic_data(
    start_date: str = "2024-01-01",
    end_date: str = "2024-01-31",
    n_locations: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic synthetic ride-hailing data for simulation.
    """
    np.random.seed(seed)
    dates = pd.date_range(start_date, end_date, freq="h")

    records = []
    for ts in dates:
        for loc_id in range(n_locations):
            base_demand = np.random.poisson(30)
            if 7 <= ts.hour <= 9 or 17 <= ts.hour <= 19:
                base_demand = int(base_demand * 2.2 + np.random.poisson(25))
            if 0 <= ts.hour <= 5:
                base_demand = max(5, int(base_demand * 0.4))

            loc_factor = 1.2 if loc_id in (0, 1) else 0.9
            demand = max(1, int(base_demand * loc_factor * np.random.uniform(0.8, 1.2)))

            base_supply = np.random.poisson(35)
            if 7 <= ts.hour <= 9 or 17 <= ts.hour <= 19:
                base_supply = int(base_supply * 1.3)
            if 0 <= ts.hour <= 5:
                base_supply = max(3, int(base_supply * 0.5))
            supply = max(1, int(base_supply * np.random.uniform(0.9, 1.1)))

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

    return pd.DataFrame(records)


def aggregate_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trip-level or location-level data to hourly buckets."""
    df = df.copy()
    df["hour_bucket"] = pd.to_datetime(df["timestamp"]).dt.floor("h")

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


def load_and_preprocess(
    raw_path: Optional[Path] = None,
    use_nyc_download: bool = True,
    data_dir: Optional[Path] = None,
    n_days_synthetic: int = 7,
) -> Tuple[pd.DataFrame, str]:
    """
    Load data: try NYC download, then local file, then synthetic.

    Returns:
        (df, data_source) where data_source is "nyc_taxi", "synthetic", etc.
    """
    if raw_path and Path(raw_path).exists():
        df = load_nyc_taxi(Path(raw_path))
        return aggregate_nyc_taxi_to_hourly(df), "nyc_taxi_local"

    if use_nyc_download and data_dir:
        data_dir = Path(data_dir)
        cached = data_dir / "raw" / "yellow_tripdata_sample.parquet"
        if cached.exists():
            df = pd.read_parquet(cached)
            return aggregate_nyc_taxi_to_hourly(_preprocess_nyc_taxi(df)), "nyc_taxi_cached"
        df = download_nyc_taxi_sample(output_dir=data_dir / "raw")
        if df is not None:
            return aggregate_nyc_taxi_to_hourly(_preprocess_nyc_taxi(df)), "nyc_taxi"

    end = pd.Timestamp("2024-01-01") + pd.Timedelta(days=n_days_synthetic)
    df = generate_synthetic_data(end_date=end.strftime("%Y-%m-%d"))
    return aggregate_to_hourly(df), "synthetic"


def _preprocess_nyc_taxi(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess NYC TLC taxi trip data to standard format."""
    pickup_col = "tpep_pickup_datetime" if "tpep_pickup_datetime" in df.columns else "pickup_datetime"
    dropoff_col = "tpep_dropoff_datetime" if "tpep_dropoff_datetime" in df.columns else "dropoff_datetime"
    df = df.rename(columns={pickup_col: "tpep_pickup_datetime", dropoff_col: "tpep_dropoff_datetime"})

    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"], errors="coerce")
    df = df.dropna(subset=["tpep_pickup_datetime", "tpep_dropoff_datetime"])
    df = df[(df["trip_distance"] > 0) & (df["fare_amount"] > 0)]
    df = df[(df["trip_distance"] < 100) & (df["fare_amount"] < 500)]
    return df


def prepare_ml_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target for demand prediction."""
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


def train_test_split_temporal(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based train/test split (no shuffle)."""
    n = len(df)
    split_idx = int(n * (1 - test_ratio))
    return df.iloc[:split_idx], df.iloc[split_idx:]
