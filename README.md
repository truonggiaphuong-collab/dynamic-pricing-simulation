# Dynamic pricing simulation for ride-hailing platforms

---

## Problem Statement

Ride-hailing platforms (Deliveroo, Uber) use **dynamic pricing** (surge pricing) to balance demand and supply. When demand exceeds driver availability, prices increase to:

- Incentivize more drivers to come online
- Reduce demand from price-sensitive customers
- Improve matching efficiency

This project simulates a complete dynamic pricing system: we **predict demand** using machine learning, apply an **economics-based pricing model**, and **simulate** outcomes over time.

---

## Dataset

We use TLC Trip Record Data with realistic temporal and spatial patterns:

| Column        | Description                          |
|---------------|--------------------------------------|
| `timestamp`   | Hourly time bucket                   |
| `hour`        | Hour of day (0–23)                    |
| `day_of_week` | Day of week (0=Monday)                |
| `location_id` | Geographic zone (0–4)                 |
| `trip_distance` | Average trip distance (miles)      |
| `demand`      | Number of ride requests               |
| `driver_supply` | Available drivers                  |
| `base_price`  | Base fare before surge                |

Demand peaks during rush hours (7–9 AM, 5–7 PM); supply is lower at night. Real NYC TLC taxi data can be integrated via `data_processing.load_and_preprocess()`.

---

## Methodology

### 1. Demand Prediction Models

We compare three models to predict demand:

| Model             | Type        | Use Case              |
|-------------------|-------------|------------------------|
| Linear Regression | Baseline    | Simple benchmark       |
| Random Forest     | Tree-based  | Non-linear patterns   |
| XGBoost           | Gradient Boosting | Best performance |

**Features:** `hour_sin`, `hour_cos`, `day_of_week`, `trip_distance`

**Evaluation:** MAE, RMSE, R²

### 2. Dynamic Pricing Model

```
price_multiplier = predicted_demand / supply
multiplier = clip(multiplier, min=1, max=3)
price = base_price × multiplier
```

- **Minimum multiplier = 1** — No discount below base fare
- **Maximum multiplier = 3** — Cap surge at 3× to limit customer backlash

### 3. Simulation Engine

- Runs pricing over multiple days (configurable)
- Tracks: demand, supply, price multiplier, revenue, rides served
- Computes: total revenue, average price, demand-served ratio

---

## Project Structure

```
dynamic-pricing-simulation/
├── README.md
├── requirements.txt
├── run_simulation.py          
├── data/
│   ├── raw/                   
│   └── processed/            
├── notebooks/
│   └── pricing_simulation.ipynb   # End-to-end demonstration
├── figures/                   # Generated plots
├── src/
│   ├── data_processing.py     # Data generation, preprocessing
│   ├── demand_model.py        # LR, RF, XGBoost demand prediction
│   ├── pricing_model.py       # Surge pricing logic
│   ├── simulation.py          # Simulation engine
│   └── visualization.py       # Plotting functions
└── docs/
    └── STEP1_REVIEW.md        # Project review
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full simulation
python run_simulation.py

# 3. Or run the Jupyter notebook
jupyter notebook notebooks/pricing_simulation.ipynb
```

---

## Simulation Results

Example output from a 7-day simulation:

| Metric               | Value    |
|----------------------|----------|
| Total Revenue        | ~$XX,XXX |
| Average Price        | ~$15–25  |
| Demand Served Ratio  | ~85–95%  |
| Best Demand Model    | XGBoost  |

### Key Insights

1. **Surge peaks during rush hours** — 7–9 AM and 5–7 PM show highest multipliers
2. **XGBoost outperforms baseline** — Captures non-linear hour-of-day effects
3. **Revenue vs multiplier** — Higher multipliers increase revenue but may reduce demand-served ratio
4. **Night hours** — Lower demand and supply; multipliers often at baseline (1.0x)

---

## Future Improvements

| Extension                  | Description                                              |
|---------------------------|----------------------------------------------------------|
| **Real data**             | Integrate NYC TLC taxi data for validation              |
| **Reinforcement Learning**| Optimize multiplier for long-term revenue               |
| **Zone-based pricing**    | Different multipliers by location (airport vs downtown) |
| **A/B testing**           | Compare pricing strategies on revenue and fairness      |
| **Causal inference**      | Estimate demand elasticity with respect to price         |

---

## License

MIT License. Data from NYC TLC is subject to their terms of use.
