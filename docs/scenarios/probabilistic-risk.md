# Scenario: Risk Management & Probabilistic Forecasting ⚖️

When making business decisions (e.g., energy bidding or supply chain logistics), a single "average" forecast is often not enough. You need to know the range of possibilities.

## 📋 Problem Description
Predict future electricity prices or demand, including the 10th and 90th percentile "risk zones".

## 🛠️ Implementation with UniTS-Hub

### Model Selection
Set `MODEL_TYPE=chronos` for robust probabilistic outputs.

### Sample Request
```json
{
  "instances": [
    {
      "history": [2.5, 2.7, 3.1, 2.9, ...],
      "metadata": {"node": "east_grid_01"}
    }
  ],
  "task": {
    "horizon": 24
  }
}
```

## 📊 Understanding Quantiles
The API returns a `quantiles` object. 
- **0.1**: The low-end "worst case" (only 10% chance it's below this).
- **0.9**: The high-end "worst case" (only 10% chance it's above this).
- **0.5 (Mean)**: The most likely scenario.

### Why this matters:
- If you are building a safety buffer, use the **90th percentile** forecast for demand.
- If you are calculating minimum revenue, use the **10th percentile** forecast.
