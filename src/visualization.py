"""
Visualizations for dynamic pricing simulation.

Creates publication-quality plots:
- Demand vs Supply over time
- Surge multiplier over time
- Price distribution
- Revenue vs price multiplier
- Hourly patterns
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_demand_vs_supply(df: pd.DataFrame, save_path: Path = None) -> None:
    """Plot demand and supply over time."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["timestamp"], df["demand"], label="Demand", alpha=0.8)
    ax.plot(df["timestamp"], df["driver_supply"], label="Supply", alpha=0.8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Count")
    ax.set_title("Demand vs Supply Over Time")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_surge_multiplier(df: pd.DataFrame, save_path: Path = None) -> None:
    """Plot surge multiplier over time."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(df["timestamp"], 1, df["surge_multiplier"], alpha=0.3)
    ax.plot(df["timestamp"], df["surge_multiplier"], color="darkorange", linewidth=1.5)
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.7, label="No surge (1.0x)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Surge Multiplier")
    ax.set_title("Surge Multiplier Over Time")
    ax.set_ylim(0.9, 3.2)
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_price_distribution(df: pd.DataFrame, save_path: Path = None) -> None:
    """Plot distribution of prices."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(df["price"], bins=30, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Price ($)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Price Distribution")

    df_plot = df.copy()
    df_plot["hour"] = df_plot["timestamp"].dt.hour
    sns.boxplot(data=df_plot, x="hour", y="price", ax=axes[1])
    axes[1].set_xlabel("Hour of Day")
    axes[1].set_ylabel("Price ($)")
    axes[1].set_title("Price by Hour of Day")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_revenue_vs_multiplier(df: pd.DataFrame, save_path: Path = None) -> None:
    """Plot revenue vs price multiplier (scatter)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["surge_multiplier"], df["revenue"], alpha=0.5, s=20)
    ax.set_xlabel("Surge Multiplier")
    ax.set_ylabel("Revenue ($)")
    ax.set_title("Revenue vs Surge Multiplier")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_counterfactual(df: pd.DataFrame, save_path: Path = None) -> None:
    """Plot surge vs no-surge revenue over time (counterfactual)."""
    if "revenue_no_surge" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["timestamp"], df["revenue"], label="With Surge", alpha=0.8)
    ax.plot(df["timestamp"], df["revenue_no_surge"], label="No Surge", alpha=0.8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Revenue ($)")
    ax.set_title("Counterfactual: Surge vs No-Surge Revenue")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_hourly_patterns(df: pd.DataFrame, save_path: Path = None) -> None:
    """Plot average demand, supply, multiplier by hour."""
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    hourly = df.groupby("hour").agg(
        demand=("demand", "mean"),
        driver_supply=("driver_supply", "mean"),
        surge_multiplier=("surge_multiplier", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax2 = ax.twinx()
    ax.bar(hourly["hour"] - 0.2, hourly["demand"], width=0.35, label="Avg Demand", alpha=0.7)
    ax.bar(hourly["hour"] + 0.2, hourly["driver_supply"], width=0.35, label="Avg Supply", alpha=0.7)
    ax2.plot(hourly["hour"], hourly["surge_multiplier"], "r-o", label="Avg Multiplier", linewidth=2)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Count")
    ax2.set_ylabel("Surge Multiplier", color="red")
    ax.set_title("Average Hourly Patterns")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_all_plots(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate all visualizations and save to figures folder."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_demand_vs_supply(df, save_path=output_dir / "demand_vs_supply.png")
    plot_surge_multiplier(df, save_path=output_dir / "surge_multiplier.png")
    plot_price_distribution(df, save_path=output_dir / "price_distribution.png")
    plot_revenue_vs_multiplier(df, save_path=output_dir / "revenue_vs_multiplier.png")
    plot_counterfactual(df, save_path=output_dir / "counterfactual_surge_vs_no_surge.png")
    plot_hourly_patterns(df, save_path=output_dir / "hourly_patterns.png")
