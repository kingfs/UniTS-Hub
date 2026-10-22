# UniTS-Hub Documentation 📚

Welcome to the UniTS-Hub documentation. This guide will help you understand how to leverage Time-Series Foundation Models for various real-world scenarios.

## 🧭 Navigation

- **[Architecture v2](architecture-v2.md)**: Target runtime, API, MCP, and release design for UniTS-Hub v2.
- **Models**:
  - **[Timestral](models/timestral.md)**: Multivariate probabilistic forecasting, covariates, missing values, and serving limits.
- **[Getting Started](getting-started/core-concepts.md)**: Core concepts of Foundation Models for time-series.
- **Scenarios**:
  - **[Retail Demand Forecasting](scenarios/retail-demand.md)**: Predicting sales and inventory needs.
  - **[Risk Management & Probabilistic Forecasting](scenarios/probabilistic-risk.md)**: Handling uncertainty with Chronos.
  - **[IoT & System Monitoring](scenarios/iot-monitoring.md)**: Real-time forecasting for sensor data.
- **[API Guide](api-guide/parameters-explained.md)**: Deep dive into API parameters and model behaviors.

## 🤔 Which Model Should I Use?

| Feature | TimesFM (Google) | Chronos (Amazon) | Timestral (Mistral) |
|---------|------------------|------------------|--------------------|
| **Best For** | High-accuracy point forecasts | Compact univariate probabilistic forecasts | Joint multivariate forecasts with covariates |
| **Quantiles** | Limited / Not default | Native support | Fixed 21-level non-crossing grid (209M) |
| **Multivariate** | No | No (in this runtime) | Yes, jointly modeled |
| **Covariates** | No | No (in this runtime) | Past-only, known-future, categorical |
| **Speed** | Fast | Very fast (especially Bolt variant) | CPU-friendly, but long contexts are costly |
| **Minimum Data**| ~32 points recommended | ~16-32 points | Short histories work; missing values allowed |
| **Context Length**| Up to 512 | Flexible | Up to 32768 |

---

> [!TIP]
> Use **TimesFM** for retail and revenue forecasting where point accuracy is critical. Use **Chronos** when you need compact univariate quantiles. Use **Timestral** when several related series must be forecast jointly or when promotions, prices, and other known-future signals should condition the forecast.
