# Step 11: Possible Extensions for Advanced Portfolio

## 1. Machine Learning Demand Prediction

**Goal**: Predict demand for the next hour using features like time, weather, events.

**Approach**:
- **Features**: hour, day_of_week, month, is_weekend, lagged demand (t-1, t-24), weather (if available)
- **Models**: XGBoost, Random Forest, or LSTM for time series
- **Output**: Predicted demand → feed into pricing model for proactive surge

**Code sketch**:
```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

features = ['hour', 'day_of_week', 'demand_lag1', 'demand_lag24']
X = df[features]
y = df['demand']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = GradientBoostingRegressor()
model.fit(X_train, y_train)
df['demand_predicted'] = model.predict(X)
```

---

## 2. Reinforcement Learning Pricing

**Goal**: Learn optimal surge multiplier to maximize long-term reward (e.g., revenue, utilization).

**Approach**:
- **State**: (demand, supply, hour, day_of_week)
- **Action**: Surge multiplier in [1.0, 3.0]
- **Reward**: Revenue = demand_served × price, or utilization = demand_served / demand
- **Algorithm**: DQN, PPO, or simpler policy gradient

**Libraries**: Stable-Baselines3, RLlib

---

## 3. Driver Allocation Optimization

**Goal**: Decide where to position idle drivers to minimize wait times and maximize matches.

**Approach**:
- **Input**: Demand by zone, supply by zone, travel times between zones
- **Output**: Recommended driver movements (e.g., from zone A to zone B)
- **Formulation**: Linear programming or heuristic (greedy matching)

**Example**:
```python
from scipy.optimize import linear_sum_assignment
# Cost matrix: cost[i,j] = -expected_revenue if driver i serves zone j
# Solve assignment problem
```

---

## 4. Zone-Based Pricing

**Goal**: Different surge multipliers by pickup zone (e.g., airport vs residential).

**Approach**:
- Aggregate demand/supply by `PULocationID`
- Apply pricing model per zone
- Compare zone-level revenue and fairness metrics

---

## 5. A/B Testing Simulation

**Goal**: Simulate impact of different pricing strategies.

**Approach**:
- Strategy A: Current model (demand/supply)
- Strategy B: Fixed 1.5x during rush hours
- Strategy C: ML-predicted demand with different caps
- Compare: revenue, utilization, customer surplus (simulated)

---

## 6. Causal Inference

**Goal**: Estimate causal effect of surge on demand (elasticity).

**Approach**:
- Instrumental variables (e.g., weather as instrument for supply)
- Difference-in-differences if you have control zones
- Regression discontinuity around surge thresholds
