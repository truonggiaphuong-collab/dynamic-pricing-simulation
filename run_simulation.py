"""
Run the full dynamic pricing simulation pipeline.

Usage:
    python run_simulation.py
"""

from pathlib import Path

from src.simulation import run_simulation, save_results
from src.visualization import generate_all_plots


def main():
    base = Path(__file__).parent
    data_dir = base / "data"
    figures_dir = base / "figures"

    print("=" * 60)
    print("Dynamic Pricing Simulation for Ride-Hailing")
    print("=" * 60)

    print("\n[1/3] Running simulation (7 days)...")
    df, summary = run_simulation(
        n_days=7,
        use_predicted_demand=True,
        use_nyc_data=True,
        data_dir=data_dir,
    )

    print("\n[2/3] Saving results...")
    save_results(df, summary, data_dir / "processed")

    print("\n[3/3] Generating visualizations...")
    generate_all_plots(df, figures_dir)

    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"Data Source:          {summary['data_source']}")
    print(f"Total Revenue:       ${summary['total_revenue']:,.2f}")
    print(f"Revenue (No Surge):   ${summary['total_revenue_no_surge']:,.2f}")
    print(f"Revenue Lift:         +{summary['revenue_lift_pct']:.1f}%")
    print(f"Average Price:       ${summary['avg_price']:.2f}")
    print(f"Average Multiplier:  {summary['avg_multiplier']:.2f}x")
    print(f"Demand Served Ratio: {summary['demand_served_ratio']:.2%}")
    print(f"Best Demand Model:   {summary['best_demand_model']}")
    print("\nModel Metrics (Test Set - Out-of-Sample):")
    for name, m in summary["model_metrics_test"].items():
        print(f"  {name}: MAE={m['MAE']:.2f}, RMSE={m['RMSE']:.2f}, R2={m['R2']:.3f}")

    print(f"\nResults saved to: {data_dir / 'processed'}")
    print(f"Figures saved to: {figures_dir}")
    print("\nDone!")


if __name__ == "__main__":
    main()
