# Step 4: Estimating Demand and Supply from the Dataset

## Overview

In ride-hailing platforms, **demand** and **supply** are typically internal metrics. With public taxi data, we use **proxies**.

## Demand Estimation

**Demand** = number of customers wanting a ride in a given time period.

### Proxies from NYC Taxi Data:
1. **Pickup count**: `COUNT(tpep_pickup_datetime)` per hour
   - Each pickup = one completed ride request
   - Assumption: completed pickups ≈ ride requests (we ignore cancelled/unfulfilled)

2. **Alternative**: Use `PULocationID` to get zone-level demand for geographic pricing

### Smoothing (optional):
- **Rolling average**: `demand_smooth = demand.rolling(3, center=True).mean()`
- **Hour-of-day average**: Fill gaps using typical hourly pattern

## Supply Estimation

**Supply** = number of drivers available to fulfill rides.

### Proxies from NYC Taxi Data:
1. **Dropoff count (lagged)**: `COUNT(tpep_dropoff_datetime)` in previous hour
   - Drivers who just finished a trip become "available"
   - Lag by 1 hour: `supply[t] = dropoffs[t-1]`

2. **Pickup count as supply**: In equilibrium, pickups ≈ dropoffs
   - Use `pickup_count` with a lag to approximate "drivers freed up"

3. **Ratio-based**: Assume supply = α × demand in baseline periods
   - Calibrate α from low-demand hours (e.g., 3–5 AM)

### Handling Zero Supply:
- Add small epsilon: `supply = max(supply, 1)` to avoid division by zero
- Or use: `supply_smooth = supply.rolling(3).mean().fillna(method='bfill')`

## Implementation

See `src/02_demand_supply_estimation.py` for the full code.
