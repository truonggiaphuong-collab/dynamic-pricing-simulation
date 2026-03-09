# Step 1 & 2: Dataset Guide and Structure

## 1. Recommended Public Datasets

### Primary Option: NYC TLC Taxi Trip Data (Recommended)
- **Source**: [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- **Alternative**: [NYC Open Data Portal](https://data.cityofnewyork.us/) or [Kaggle NYC Taxi](https://www.kaggle.com/datasets)
- **Why**: Real-world ride-hailing data with timestamps, locations, distances, and fares
- **Format**: CSV (monthly files)
- **Size**: ~1-2GB per month for Yellow Taxi

### Alternative Options:
| Dataset | Source | Use Case |
|---------|--------|----------|
| **Chicago Taxi Trips** | data.cityofchicago.org | Similar structure to NYC |
| **Uber Movement** | movement.uber.com | Aggregated travel times (anonymized) |
| **DiDi GAIA** | gaia.didichuxing.com | Chinese ride-hailing (if available) |
| **Citi Bike NYC** | citibikenyc.com/system-data | Bike-sharing demand patterns |

---

## 2. Dataset Structure for Dynamic Pricing

Your processed dataset should have the following structure:

| Variable | Type | Description | How to Derive |
|----------|------|-------------|---------------|
| **timestamp** | datetime | Time bucket (e.g., hourly) | Aggregate from `tpep_pickup_datetime` |
| **demand** | int/float | Number of ride requests in period | Count of pickups per time bucket |
| **driver_supply** | int/float | Available drivers/rides completed | Count of dropoffs (proxy) or trips started |
| **trip_distance** | float | Average trip distance (miles) | Mean of `trip_distance` per bucket |
| **base_price** | float | Base fare before surge | `fare_amount` or fixed base (e.g., $2.50) |
| **total_amount** | float | Actual charged amount | From `total_amount` column |
| **zone_id** | int | Geographic zone (optional) | `PULocationID` for zone-based pricing |

### Variable Definitions:

- **timestamp**: The time period for aggregation (e.g., 2024-01-15 14:00:00 for 2-3 PM)
- **demand**: Number of customers requesting rides—higher during rush hours, events
- **driver_supply**: Proxy for available drivers; in real platforms this is internal data
- **trip_distance**: Affects base fare; used for distance-based pricing
- **base_price**: Minimum fare (e.g., $2.50 flag drop + $0.50/mile)
- **total_amount**: What customers actually paid; used to validate pricing model

### NYC Yellow Taxi Raw Columns (Reference):
- `tpep_pickup_datetime`, `tpep_dropoff_datetime`
- `PULocationID`, `DOLocationID`
- `trip_distance`, `fare_amount`, `total_amount`
- `passenger_count`, `RatecodeID`
