# Dynamic Pricing Simulation for Ride-Hailing Platforms

A portfolio project demonstrating **data analysis**, **machine learning**, **economics-based pricing logic**, and **data visualization** for graduate school applications.

---

## Problem Statement

Ride-hailing platforms (e.g., Grab, Uber) use **dynamic pricing** (surge pricing) to balance demand and supply. When demand exceeds driver availability, prices increase to:
- Incentivize more drivers to come online
- Reduce demand from price-sensitive customers
- Improve matching efficiency

This project simulates a simple dynamic pricing model using public taxi data, estimates demand and supply, and visualizes pricing outcomes over time.

---

## Methodology

### 1. Data Source
- **Primary**: [NYC TLC Taxi Trip Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- **Alternative**: Synthetic sample data (included for quick testing)

### 2. Data Structure
| Variable       | Description                                      |
|----------------|--------------------------------------------------|
| `timestamp`    | Time bucket (hourly)                             |
| `demand`       | Number of ride requests (pickup count)           |
| `driver_supply`| Available drivers (dropoff count as proxy)       |
| `trip_distance`| Average trip distance (miles)                    |
| `base_price`   | Base fare before surge                           |

### 3. Demand & Supply Estimation
- **Demand**: Count of pickups per hour
- **Supply**: Count of dropoffs (lagged) as proxy for drivers becoming available

### 4. Pricing Model
```
surge_multiplier = demand / supply
surge_multiplier = clip(surge_multiplier, min=1, max=3)
simulated_price = base_price × surge_multiplier
```

Constraints: multiplier ∈ [1, 3] (no discount below base, cap at 3×)

---

## Project Structure

```
dynamic-pricing-simulation/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/              # Place NYC taxi CSV here
│   └── processed/        # Aggregated & simulated outputs
├── src/
│   ├── 01_data_preprocessing.py
│   ├── 02_demand_supply_estimation.py
│   ├── 03_pricing_model.py
│   ├── 04_run_simulation.py
│   └── 05_visualizations.py
├── outputs/
│   └── figures/          # Generated plots
└── docs/
    ├── 01_DATASET_GUIDE.md
    └── 02_DEMAND_SUPPLY_ESTIMATION.md
```

---

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run simulation (uses synthetic data if no raw data)
python src/04_run_simulation.py

# 4. Generate visualizations
python src/05_visualizations.py
```

### Using Real NYC Taxi Data
1. Download from [NYC TLC](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
2. Place `yellow_tripdata_YYYY-MM.csv` in `data/raw/`
3. Edit `01_data_preprocessing.py`: remove `nrows=100_000` for full data
4. Run `python src/01_data_preprocessing.py`
5. Run `python src/02_demand_supply_estimation.py`
6. Run `python src/03_pricing_model.py` (or use `04_run_simulation.py` with `use_sample=False`)

---

## Results

The simulation produces:
- **Demand vs Supply** over time
- **Surge multiplier** over time (1.0x = no surge, up to 3.0x)
- **Price distribution** and hourly patterns

Example findings:
- Surge peaks during rush hours (7–9 AM, 5–7 PM)
- Night hours show lower demand and supply
- Price distribution reflects time-of-day patterns

---

## Conclusion

This project demonstrates:
- **Data engineering**: Preprocessing and aggregating raw trip data
- **Economics**: Demand-supply equilibrium and surge pricing logic
- **Coding**: Modular Python pipeline with clear structure
- **Visualization**: Publication-ready plots with matplotlib/seaborn

---

## Possible Extensions

| Extension                    | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| **ML Demand Prediction**    | Use XGBoost/LSTM to predict demand from weather, events, historical patterns |
| **Reinforcement Learning**  | Train an RL agent to optimize multiplier for revenue or social welfare      |
| **Driver Allocation**       | Optimize driver positioning across zones to reduce wait times               |
| **Zone-Based Pricing**      | Different multipliers by `PULocationID` (e.g., airport vs downtown)         |
| **A/B Testing Simulation**  | Compare pricing strategies on revenue, utilization, customer satisfaction  |

---

## License

MIT License. Data from NYC TLC is subject to their terms of use.
