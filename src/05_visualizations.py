"""
Step 8: Visualizations for Dynamic Pricing Simulation

Creates:
- Demand vs Supply over time
- Surge multiplier over time
- Price distribution
- Additional: demand/supply ratio, hourly patterns

Usage:
    python src/05_visualizations.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# Set style for publication-quality plots
sns.set_style("whitegrid")
sns.set_palette("husl")


def load_simulation_data() -> pd.DataFrame:
    """Load pricing simulation results."""
    data_dir = Path(__file__).parent.parent / 'data'
    path = data_dir / 'processed' / 'pricing_simulation.csv'

    if not path.exists():
        raise FileNotFoundError(
            "Run 04_run_simulation.py first to generate pricing_simulation.csv"
        )

    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def plot_demand_vs_supply(df: pd.DataFrame, save_path: Path = None, show: bool = False):
    """Plot demand and supply over time."""
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(df['timestamp'], df['demand'], label='Demand (ride requests)', alpha=0.8)
    ax.plot(df['timestamp'], df['driver_supply'], label='Supply (available drivers)', alpha=0.8)

    ax.set_xlabel('Time')
    ax.set_ylabel('Count')
    ax.set_title('Demand vs Supply Over Time')
    ax.legend(loc='upper right')
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_surge_multiplier_over_time(df: pd.DataFrame, save_path: Path = None, show: bool = False):
    """Plot surge multiplier over time."""
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.fill_between(df['timestamp'], 1, df['surge_multiplier'], alpha=0.3)
    ax.plot(df['timestamp'], df['surge_multiplier'], color='darkorange', linewidth=1.5)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='No surge (1.0x)')

    ax.set_xlabel('Time')
    ax.set_ylabel('Surge Multiplier')
    ax.set_title('Surge Multiplier Over Time')
    ax.set_ylim(0.9, 3.2)
    ax.legend(loc='upper right')
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_price_distribution(df: pd.DataFrame, save_path: Path = None, show: bool = False):
    """Plot distribution of simulated prices."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    axes[0].hist(df['simulated_price'], bins=30, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Simulated Price ($)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Simulated Prices')

    # Box plot by hour of day
    df_plot = df.copy()
    df_plot['hour'] = df_plot['timestamp'].dt.hour
    sns.boxplot(data=df_plot, x='hour', y='simulated_price', ax=axes[1])
    axes[1].set_xlabel('Hour of Day')
    axes[1].set_ylabel('Simulated Price ($)')
    axes[1].set_title('Price Distribution by Hour of Day')
    axes[1].tick_params(axis='x', rotation=0)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_demand_supply_ratio(df: pd.DataFrame, save_path: Path = None, show: bool = False):
    """Plot demand/supply ratio over time."""
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(df['timestamp'], df['demand_supply_ratio'], color='purple', alpha=0.8)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='Equilibrium (D/S=1)')

    ax.set_xlabel('Time')
    ax.set_ylabel('Demand / Supply Ratio')
    ax.set_title('Demand-Supply Ratio Over Time')
    ax.legend(loc='upper right')
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_hourly_patterns(df: pd.DataFrame, save_path: Path = None, show: bool = False):
    """Plot average demand, supply, and multiplier by hour of day."""
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour

    hourly = df.groupby('hour').agg({
        'demand': 'mean',
        'driver_supply': 'mean',
        'surge_multiplier': 'mean'
    }).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax2 = ax.twinx()

    ax.bar(hourly['hour'] - 0.2, hourly['demand'], width=0.35, label='Avg Demand', alpha=0.7)
    ax.bar(hourly['hour'] + 0.2, hourly['driver_supply'], width=0.35, label='Avg Supply', alpha=0.7)
    ax2.plot(hourly['hour'], hourly['surge_multiplier'], 'r-o', label='Avg Surge Multiplier', linewidth=2)

    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Count')
    ax2.set_ylabel('Surge Multiplier', color='red')
    ax.set_title('Average Hourly Patterns: Demand, Supply, and Surge')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def main(show: bool = False):
    """Generate all visualizations.
    
    Args:
        show: If True, display plots interactively. Default False (save only).
    """
    df = load_simulation_data()

    output_dir = Path(__file__).parent.parent / 'outputs' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating visualizations...")

    plot_demand_vs_supply(df, save_path=output_dir / '01_demand_vs_supply.png', show=show)
    plot_surge_multiplier_over_time(df, save_path=output_dir / '02_surge_multiplier_over_time.png', show=show)
    plot_price_distribution(df, save_path=output_dir / '03_price_distribution.png', show=show)
    plot_demand_supply_ratio(df, save_path=output_dir / '04_demand_supply_ratio.png', show=show)
    plot_hourly_patterns(df, save_path=output_dir / '05_hourly_patterns.png', show=show)

    print(f"All figures saved to {output_dir}")


if __name__ == '__main__':
    main()
