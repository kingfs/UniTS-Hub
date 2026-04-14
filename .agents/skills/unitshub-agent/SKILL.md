---
name: unitshub-agent
description: Use UniTS-Hub as a capability-first time-series tool via REST or MCP. Trigger when the task involves TimesFM, Chronos-2, Kronos, choosing among their forecasting capabilities, or constructing valid UniTS-Hub requests for an AI agent.
---

# UniTS-Hub Agent

Use this skill when an agent needs to call a UniTS-Hub deployment.

## Model selection

- Use `TimesFM` for univariate point forecasting with compact inputs.
- Use `Chronos-2` for quantile forecasting, multivariate targets, or requests that include covariates.
- Use `Kronos` for financial candle data such as OHLCV and sampled path generation.

## Preferred protocol

- Prefer MCP when the client supports JSON-RPC tool calls.
- Fallback to REST when the client only supports HTTP APIs.
- Discover the active model before invoking it. Do not assume a deployment serves all three models at once.

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
   - `invoke_model`

Example `tools/call` payload:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "invoke_model",
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

- `TimesFM`: `input.series[].target` is a numeric array.
- `Chronos-2`: `input.series[].target` may be univariate or multivariate. Add `past_covariates` and `future_covariates` only when the deployment advertises the covariate task.
- `Kronos`: `input.series[].candles` is an array of OHLCV-like records. Include `timestamp` on each candle when available.

## Legacy note

- `/predict` and `/predict/csv` are compatibility endpoints for older workflow tools.
- New agent integrations should use `/models/current/invoke` or `/mcp`.
