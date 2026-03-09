"""
Step 7: Run Full Pricing Simulation Across Time

Orchestrates the full pipeline:
1. Load/preprocess data
2. Estimate demand/supply
3. Apply pricing model
4. Output results for visualization

Simulates pricing at hourly granularity.

Usage:
    python src/04_run_simulation.py
"""

import importlib.util
import pandas as pd
from pathlib import Path


def _load_module(name: str, filename: str):
    """Load module from file (handles 01_ prefix in filenames)."""
    base = Path(__file__).parent
    spec = importlib.util.spec_from_file_location(name, base / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_full_simulation(use_sample: bool = True) -> pd.DataFrame:
    """
    Run the complete dynamic pricing simulation.

    Args:
        use_sample: If True, use synthetic data. If False, load from data/processed/
    """
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    (data_dir / 'processed').mkdir(exist_ok=True)

    data_prep = _load_module('data_preprocessing', '01_data_preprocessing.py')
    est = _load_module('demand_supply_estimation', '02_demand_supply_estimation.py')
    pricing = _load_module('pricing_model', '03_pricing_model.py')

    if use_sample:
        print("Using synthetic sample data...")
        df = data_prep.create_sample_data()
    else:
        input_path = data_dir / 'processed' / 'hourly_aggregated.csv'
        if not input_path.exists():
            raise FileNotFoundError(
                "Run 01_data_preprocessing.py first, or set use_sample=True"
            )
        df = pd.read_csv(input_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    df = est.estimate_demand_supply(df)
    df = est.add_demand_supply_ratio(df)

    config = pricing.PricingConfig(min_multiplier=1.0, max_multiplier=3.0)
    df = pricing.apply_pricing_vectorized(df, config)

    output_path = data_dir / 'processed' / 'pricing_simulation.csv'
    df.to_csv(output_path, index=False)
    print(f"Simulation complete. Results saved to {output_path}")

    return df


if __name__ == '__main__':
    df = run_full_simulation(use_sample=True)
    print(f"\nTotal periods: {len(df)}")
    print(f"Surge multiplier range: [{df['surge_multiplier'].min():.2f}, {df['surge_multiplier'].max():.2f}]")
