# Scenario: IoT & System Monitoring 📟

Monitoring thousands of sensors or server metrics requires fast, real-time forecasting to set dynamic thresholds.

## 📋 Problem Description
Predict CPU usage for the next hour to proactively scale up instances before a bottleneck occurs.

## 🛠️ Implementation with UniTS-Hub

### Model Selection
Both models work well, but **Chronos-Bolt** (if configured) or standard Chronos offers high inference speed for high-throughput sensor data.

### Sample Request
```json
{
  "instances": [
    {"history": [45.2, 48.1, 52.0, 50.5, 49.0], "metadata": {"server": "app_01"}},
    {"history": [10.1, 12.5, 11.2, 10.8, 11.0], "metadata": {"server": "db_01"}}
  ],
  "task": {
    "horizon": 6 // Next hour (assuming 10min intervals)
  },
  "parameters": {
    "freq": "10T"
  }
}
```

## 💡 Best Practices for IoT
- **Dynamic Thresholds**: Use the upper quantile (e.g., 0.95) as an alert threshold. If actual usage crosses the *predicted* upper bound, it's a true anomaly.
- **Short Context**: For high-frequency data, sometimes a short context (last 50-100 points) is enough to capture immediate trends.
- **Frequency (`freq`)**: Always specify the data frequency (e.g., 'T' for minutes, 'H' for hours) to help the model align with potential daily/weekly seasonality.
