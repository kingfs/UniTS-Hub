# Scenario: Retail Demand Forecasting 🛒

In retail, understocking leads to lost sales, and overstocking leads to waste. TimesFM is particularly strong at capturing complex retail patterns.

## 📋 Problem Description
Predict the next 14 days of sales for a SKU based on the last 90 days of history.

## 🛠️ Implementation with UniTS-Hub

### Model Selection
Set `MODEL_TYPE=timesfm` for high-accuracy point forecasting.

### Sample Request
```json
{
  "instances": [
    {
      "history": [10, 15, 12, 14, 20, 25, ...], // 90 days of history
      "metadata": {"sku_id": "laptop_x1"}
    }
  ],
  "task": {
    "horizon": 14
  }
}
```

## 💡 Best Practices for Retail
- **Data Density**: Provide at least 30 points. If you have weekly data, provide at least 30 weeks.
- **Seasonality**: Foundation models are great at spotting weekly patterns (e.g., weekend peaks) without you telling them.
- **Batching**: If you have 1000 SKUs, send them in batches (e.g., 50 per request) to maximize throughput.
