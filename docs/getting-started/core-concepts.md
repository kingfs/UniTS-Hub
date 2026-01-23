# Core Concepts: Foundation Models for Time-Series

## 🌟 What are Foundation Models?

Just as LLMs (Large Language Models) like GPT-4 are trained on vast amounts of text to understand language, **Time-Series Foundation Models** are trained on trillions of time-points across diverse domains (finance, weather, retail, energy).

### Key Advantages:

1. **Zero-Shot Inference**: You can get accurate forecasts for your data WITHOUT training a model on it. The model already knows "what a trend looks like" and "how seasonality works."
2. **Reduced Pipeline Complexity**: No more expensive feature engineering, scaling, or model selection per time series.
3. **Robustness**: Because they've seen so much data, they are less likely to overfit on small, noisy datasets.

## 📈 Univariate vs Multivariate

Most current foundation models (including TimesFM and Chronos) are **Univariate**. This means they look at the history of one variable to predict its future.
- **Example**: Using past "Sales" to predict "Future Sales."
- **Note**: While they don't ingest "Weather" or "Holiday" features directly in a multivariate way, their pre-training often captures these patterns implicitly (e.g., they recognice a 7-day or 365-day cycle automatically).

## 🎯 Accuracy and Context

- **Context Length**: The number of historical points you provide. Foundation models typically benefit from longer context, up to their limit (e.g., 512 for TimesFM).
- **Horizon**: How far into the future you want to predict. Accuracy naturally degrades as you predict further out.
