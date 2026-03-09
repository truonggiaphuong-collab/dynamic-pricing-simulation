"""
Run the full dynamic pricing simulation pipeline.

Usage:
    python run_all.py
"""

import importlib.util
from pathlib import Path

def load_module(name: str, filepath: Path):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

if __name__ == '__main__':
    base = Path(__file__).parent

    print("=" * 50)
    print("Step 1: Running pricing simulation...")
    print("=" * 50)
    run_sim = load_module("run_sim", base / "src" / "04_run_simulation.py")
    df = run_sim.run_full_simulation(use_sample=True)

    print("\n" + "=" * 50)
    print("Step 2: Generating visualizations...")
    print("=" * 50)
    viz = load_module("viz", base / "src" / "05_visualizations.py")
    viz.main()

    print("\nDone!")
