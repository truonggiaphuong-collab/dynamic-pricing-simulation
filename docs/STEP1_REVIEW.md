# STEP 1 — Project Review

## What Is Currently Implemented

| Component | Status | Notes |
|-----------|--------|-------|
| **Data pipeline** | Partial | `create_sample_data()` generates synthetic hourly data; raw NYC taxi loading exists but requires external file |
| **Demand/supply** | Basic | Uses pickup count = demand, lagged dropoff = supply; no ML prediction |
| **Pricing model** | Yes | `price_multiplier = demand/supply` with [1, 3] constraints |
| **Simulation** | Basic | Runs pricing over time; no revenue/rides-served tracking |
| **Visualizations** | Yes | 5 plots: demand vs supply, surge over time, price dist, ratio, hourly patterns |
| **Structure** | Numbered scripts | `01_` to `05_`; uses importlib for module loading |

## What Is Missing

- **Demand prediction models**: No Linear Regression, Random Forest, or XGBoost
- **Model evaluation**: No MAE, RMSE, R² metrics
- **Simulation metrics**: No revenue, rides served, demand-served ratio
- **Jupyter notebook**: No end-to-end demonstration
- **Revenue vs multiplier plot**: Not in current visualizations
- **Location/zone support**: No `location_id` in data structure
- **Professional structure**: `notebooks/`, `figures/` (vs `outputs/figures`), clean `src/` modules

## What Needs Improvement

- **Modularity**: Replace numbered scripts with named modules (`data_processing`, `demand_model`, etc.)
- **Data schema**: Add `hour`, `day_of_week`, `location_id` for ML features
- **Docstrings**: Some functions lack full docstrings
- **README**: Convert to mini research format (Problem, Methodology, Results, Insights)
- **Dependencies**: Add `xgboost` for demand prediction
