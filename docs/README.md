# UniTS-Hub Documentation 📚

Welcome to the UniTS-Hub documentation. This guide will help you understand how to leverage Time-Series Foundation Models for various real-world scenarios.

## 🧭 Navigation

- **[Getting Started](getting-started/core-concepts.md)**: Core concepts of Foundation Models for time-series.
- **Scenarios**:
  - **[Retail Demand Forecasting](scenarios/retail-demand.md)**: Predicting sales and inventory needs.
  - **[Risk Management & Probabilistic Forecasting](scenarios/probabilistic-risk.md)**: Handling uncertainty with Chronos.
  - **[IoT & System Monitoring](scenarios/iot-monitoring.md)**: Real-time forecasting for sensor data.
- **[API Guide](api-guide/parameters-explained.md)**: Deep dive into API parameters and model behaviors.

## 🤔 Which Model Should I Use?

| Feature | TimesFM (Google) | Chronos (Amazon) |
|---------|------------------|------------------|
| **Best For** | High-accuracy point forecasts | Probabilistic ranges & uncertainty |
| **Quantiles** | Limited / Not default | Native support (Full distribution) |
| **Speed** | Fast | Very fast (especially Bolt variant) |
| **Minimum Data**| ~32 points recommended | ~16-32 points |
| **Context Length**| Up to 512 | Flexible |

---

> [!TIP]
> Use **TimesFM** for retail and revenue forecasting where point accuracy is critical. Use **Chronos** when you need to know the "worst-case" or "best-case" scenarios via quantiles.
