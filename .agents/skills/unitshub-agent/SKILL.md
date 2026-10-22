---
name: unitshub-agent
description: Use UniTS-Hub as a capability-first time-series tool via REST or MCP. Trigger when the task involves TimesFM, Chronos-2, Kronos, Timestral, choosing among their forecasting capabilities, or constructing valid UniTS-Hub requests for an AI agent.
---

# UniTS-Hub Agent

Use this skill when an agent needs to call a UniTS-Hub deployment.

## Model selection

- Use `TimesFM` for univariate point forecasting with compact inputs.
- Use `Chronos-2` for compact univariate quantile forecasting.
- Use `Timestral` for jointly forecast multivariate series, for covariates (past-only, known-future, categorical), or when missing observations must be modeled.
- Use `Kronos` for financial candle data such as OHLCV and sampled path generation.

The active deployment serves exactly one model family. Always read `GET /models/current` before choosing a task; the task list is the source of truth.

## Preferred protocol

- Prefer MCP when the client supports JSON-RPC tool calls.
- Fallback to REST when the client only supports HTTP APIs.
- Discover the active model and its task schemas before invoking it. Do not assume a deployment serves every model family at once.

## REST workflow

1. Call `GET /models/current`.
2. Call `GET /models/current/schema` or `GET /models/current/tasks/{task}/schema`.
3. Call `POST /models/current/invoke` with:

```json
{
  "task": "forecast_point",
  "input": {
    "series": [{"target": [1.0, 2.0, 3.0]}],
    "horizon": 12
  }
}
```

## MCP workflow

1. Send `initialize`.
2. Send `tools/list`.
3. Use `tools/call` with:
   - `get_current_model`
   - `get_model_schema`
   - `get_task_schema`
   - `invoke_task`

Example `tools/call` payload:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "invoke_task",
    "arguments": {
      "task": "forecast_quantile",
      "input": {
        "series": [{"target": [1, 2, 3, 4]}],
        "horizon": 8,
        "quantiles": [0.1, 0.5, 0.9]
      }
    }
  }
}
```

## Input conventions

- `TimesFM`: `input.series[].target` is a numeric array; task `forecast_point`.
- `Chronos-2`: `input.series[].target` is a numeric array; task `forecast_quantile`.
- `Timestral`:
  - For one univariate series, `/timestral/forecast` takes the same payload as `/chronos/forecast` and returns the same `{"mean", "quantiles"}` shape. Only the route differs.
  - `input.series[].target` is a `(variates, time)` matrix for batch/multivariate requests via `/timestral/forecast-batch` or the generic invoke task. A flat array is treated as one variate.
  - Use task `forecast_quantile` for targets only and `forecast_with_covariates` when covariates are present.
  - Covariate fields: `past_covariates` and `categorical_past_covariates` cover `time` steps; `known_covariates` and `categorical_known_covariates` cover `time + horizon` steps (the last `horizon` values are the future).
  - Use JSON `null` for missing observations; do not substitute zero.
  - `quantiles` must be a subset of `metadata.available_quantiles` from `GET /models/current`.
  - `mean` is the median. The response includes `variate_count`; multivariate values are nested `[variates][horizon]`.
  - The runtime has no sampled paths; do not request `generate_paths`.
- `Kronos`: `input.series[].candles` is an array of OHLCV-like records. Include `timestamp` on each candle when available.

## Legacy note

- `/predict` and `/predict/csv` are compatibility endpoints for older workflow tools.
- New agent integrations should use `/models/current/invoke` or `/mcp`.
