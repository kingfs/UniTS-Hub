# API Guide: Parameters Explained 🔍

This guide details the common parameters used in UniTS-Hub and how they influence the models.

## 📥 Request Parameters

### `instances` (List)
Each instance represents one time series.
- `history`: List of numbers. The more historical data you provide, the better. Recommended minimum: 32.
- `metadata`: Any dictionary. This is returned back to you to help identify which series the forecast belongs to.

### `task.horizon` (Int)
The number of future steps to predict. 
- **Short Horizon (1-10)**: Very high accuracy.
- **Medium Horizon (10-50)**: Good for planning.
- **Long Horizon (50+)**: Accuracy drops significantly; use probabilistic results to understand uncertainty.

### `parameters` (Dict)
- `freq` (String): The frequency of your data using Pandas-style offsets.
  - Examples: `H` (Hourly), `D` (Daily), `W` (Weekly), `T` (Minutely), `M` (Monthly).
  - *Why it matters*: TimesFM uses `freq` to understand seasonality (e.g., "Is this a weekend?").
- `num_samples` (Int): Used by Chronos for probabilistic forecasting.
  - Higher values = better quantile accuracy but slightly slower inference. Default is usually 20.

## 📤 Response Parameters

### `forecasts` (List)
- `mean`: The predicted values.
- `quantiles`: A map of percentile keys (e.g., "0.1", "0.9") to their predicted values.

---

## 🎛️ Handling Scalability
Foundation models are larger than traditional models (like AutoARIMA).
1. **GPU Acceleration**: CUDA is highly recommended for production.
2. **Concurrency**: UniTS-Hub can handle multiple requests, but for very high loads, scale out the Docker containers.
3. **Batching**: Always prefer sending multiple series in one `instances` list rather than many separate API calls.
