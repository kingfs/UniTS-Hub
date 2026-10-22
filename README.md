# UniTS-Hub

Capability-first serving for time-series foundation models.

UniTS-Hub v2 keeps the original single-model deployment model, but replaces the old predict-only interface with a model-capability API designed for AI agents. The service now targets four model families:

- `TimesFM 3.0` (and 2.5-compatible checkpoints) for univariate point forecasting
- `Chronos-2` for compact univariate quantile forecasting
- `Kronos` for financial OHLCV forecasting and sampled path generation
- `Timestral` (Mistral, 209M and smaller checkpoints) for multivariate probabilistic forecasting with past-only and known-future covariates

It exposes three integration surfaces:

- REST API for Dify, LangChain, Flowise, and generic HTTP clients
- MCP endpoint powered by the official [`modelcontextprotocol/python-sdk`](https://github.com/modelcontextprotocol/python-sdk)
- A repo-local Codex skill at [`.agents/skills/unitshub-agent/SKILL.md`](/Users/kingfs/go/src/github.com/kingfs/UniTS-Hub/.agents/skills/unitshub-agent/SKILL.md)

## Core design

- One container still serves one loaded model at a time.
- Agents discover model capabilities at runtime instead of relying on static docs.
- Task schemas are model-aware and exposed explicitly.
- Quantile models share one univariate request/response contract: `Chronos-2` and `Timestral` use the same payload shape, so switching models changes only the route.
- Legacy `/predict` remains for backward compatibility.

## API surface

### Discover the current model

`GET /models/current`

Example response:

```json
{
  "id": "chronos",
  "name": "Chronos-2",
  "version": "2",
  "input_modes": ["univariate", "multivariate", "covariates"],
  "output_modes": ["quantile_forecast"],
  "tasks": [
    {
      "name": "forecast_quantile",
      "title": "Quantile Forecast"
    }
  ]
}
```

### Fetch schemas

- `GET /models/current/schema`
- `GET /models/current/tasks/{task}/schema`

### Invoke a capability

`POST /models/current/invoke`

Example for `TimesFM`:

```json
{
  "task": "forecast_point",
  "input": {
    "series": [
      {
        "target": [10.5, 12.1, 11.8, 13.2, 12.9],
        "item_id": "sensor_01"
      }
    ],
    "horizon": 5,
    "frequency": "1h"
  }
}
```

Example for `Chronos-2`:

```json
{
  "task": "forecast_quantile",
  "input": {
    "series": [
      {
        "item_id": "retail-sku-42",
        "target": [120, 125, 118, 131, 135]
      }
    ],
    "horizon": 7,
    "quantiles": [0.1, 0.5, 0.9]
  }
}
```

For a single univariate series, `Timestral` uses the **same request and response contract as `Chronos-2`**. Only the model-specific route differs:

```bash
curl -X POST http://localhost:8000/timestral/forecast \
  -H "Authorization: Bearer unitshub-secret" \
  -H "Content-Type: application/json" \
  -d '{
    "series": [120, 125, 118, 131, 135, 133, 140],
    "horizon": 7,
    "quantiles": [0.1, 0.5, 0.9]
  }'
```

```json
{
  "mean": [141.2, 142.0, 142.6, 143.1, 143.5, 143.9, 144.2],
  "quantiles": {
    "0.1": [139.8, 140.4, 140.9, 141.3, 141.7, 142.0, 142.3],
    "0.5": [141.2, 142.0, 142.6, 143.1, 143.5, 143.9, 144.2],
    "0.9": [142.7, 143.6, 144.3, 144.9, 145.4, 145.9, 146.3]
  }
}
```

For joint multivariate or batched forecasts, use `POST /timestral/forecast-batch` (or the generic invoke with the same task):

```json
{
  "task": "forecast_quantile",
  "input": {
    "series": [
      {
        "item_id": "store-42",
        "target": [
          [120, 125, 118, 131, 135, 133, 140],
          [9.9, 9.8, 10.1, 10.0, 10.3, 10.2, 10.4]
        ],
        "variate_names": ["demand", "price"]
      }
    ],
    "horizon": 7,
    "quantiles": [0.1, 0.5, 0.9]
  }
}
```

For covariates, use `POST /timestral/forecast-with-covariates` (task `forecast_with_covariates`):

```json
{
  "task": "forecast_with_covariates",
  "input": {
    "series": [
      {
        "item_id": "store-42",
        "target": [[120, 125, 118, 131, 135, 133, 140]],
        "past_covariates": [[0, 1, 0, 0, 1, 1, 0]],
        "known_covariates": [[0, 1, 1, 0, 0, 1, 1, 0, 1, 0]],
        "categorical_known_covariates": [[2, 2, 2, 2, 2, 2, 2, 1, 1, 1]]
      }
    ],
    "horizon": 3,
    "quantiles": [0.1, 0.5, 0.9]
  }
}
```

`target` is a `(variates, time)` matrix; a flat array such as `[120, 125, 118]` is accepted as one variate. `known_covariates` cover `time + horizon` steps: the last `horizon` values are the known future. `null` marks a missing observation. Full details are in [docs/models/timestral.md](docs/models/timestral.md).

Example for `Kronos`:

```json
{
  "task": "generate_paths",
  "input": {
    "symbol": "AAPL",
    "candles": [
      {
        "timestamp": "2026-04-10T00:00:00Z",
        "open": 190.1,
        "high": 191.4,
        "low": 188.7,
        "close": 189.8,
        "volume": 51230000
      }
    ],
    "horizon": 5,
    "num_samples": 4
  }
}
```

`curl` 示例需要显式带 `Content-Type: application/json`。服务端现在也会兼容常见的 `curl -d` 省略该头的写法，但仍建议始终带上：

```bash
curl -X POST http://localhost:8000/models/current/invoke \
  -H "Authorization: Bearer unitshub-secret" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "forecast_ohlcv",
    "input": {
      "symbol": "AAPL",
      "candles": [
        {
          "timestamp": "2026-04-10T00:00:00Z",
          "open": 190.1,
          "high": 191.4,
          "low": 188.7,
          "close": 189.8,
          "volume": 51230000
        }
      ],
      "horizon": 5
    }
  }'
```

## Docker

Single image:

```bash
docker run --rm \
  -p 8000:8000 \
  -e MODEL_TYPE=timesfm \
  -e API_KEY=unitshub-secret \
  kingfs/unitshub:timesfm-latest
```

Compose profile switching:

1. Create a local `.env` file and set `COMPOSE_PROFILES=kronos`
2. Optionally set `API_KEY=unitshub-secret`
3. Start the selected profile with `docker compose -f docker-compose.example.yml up -d`

The example compose file keeps one service per model image, and `COMPOSE_PROFILES` decides which one starts.

## MCP endpoint

`POST /mcp`

The server uses the official MCP Python SDK in stateless Streamable HTTP mode. Available tools:

- `get_current_model`
- `get_model_schema`
- `get_task_schema`
- `invoke_task`

Example:

```bash
curl -X POST http://localhost:8000/mcp/ \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "Authorization: Bearer your-secret-key" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "invoke_task",
      "arguments": {
        "task": "forecast_point",
        "input": {
          "history": [1, 2, 3, 4],
          "horizon": 3,
          "frequency": "auto"
        }
      }
    }
  }'
```

The SDK handles MCP protocol details, so UniTS-Hub only defines tool behavior.

## Legacy compatibility

The original endpoints still exist:

- `POST /predict`
- `POST /predict/csv`

These are marked as compatibility interfaces. New agent integrations should prefer `/models/current/invoke` or `/mcp`.

## Configuration

| Variable | Description | Default |
| --- | --- | --- |
| `MODEL_TYPE` | `timesfm`, `chronos`, `kronos`, or `timestral` | `chronos` |
| `MODELS_DIR` | Base directory containing model weights | `/app/models` |
| `API_KEY` | Bearer token used by API and MCP | `unitshub-secret` |
| `KRONOS_TOKENIZER_PATH` | Optional local tokenizer path for Kronos | unset |
| `KRONOS_RUNTIME_PATH` | Location of the official Kronos source runtime inside the container | `/opt/kronos-runtime` |
| `TIMESTRAL_MAX_CONTEXT` | Optional context cap in time steps (most recent steps are kept) | checkpoint default (`32768` for 209M) |

## Model assets

Download bundled model assets:

```bash
python3 scripts/download_models.py
```

Download a specific model:

```bash
python3 scripts/download_models.py --model kronos
```

`kronos` downloads both the model weights and the tokenizer repository.

`timestral` downloads `config.json`, `model.safetensors`, and the license from `mistralai/Timestral-209M`.

## Local development

```bash
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# For MODEL_TYPE=timestral, install the optional runtime extra instead:
uv sync --extra timestral
uv run --extra timestral uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Quality gates (the same commands run in CI):

```bash
uvx ruff@0.17.1 check app tests scripts
uvx ruff@0.17.1 format --check app tests scripts
uv run --extra timestral pytest
```

Real-checkpoint Timestral integration tests are opt-in because the 209M weights are not
downloaded in CI:

```bash
TIMESTRAL_REAL_MODEL=$PWD/models/timestral uv run --extra timestral pytest tests/test_timestral_real_model.py
```

## Deployed API smoke test

After you start one model container, you can validate the live service with:

```bash
python3 scripts/api_smoke_test.py --base-url http://localhost:8000 --api-key unitshub-secret
```

The script first calls `/models/current`, detects whether the service is running `timesfm`, `chronos`, `kronos`, or `timestral`, then sends model-specific sample payloads to `/models/current/invoke` and the direct model route:

- `timesfm`: `/timesfm/forecast`
- `chronos`: `/chronos/forecast`
- `kronos`: `/kronos/forecast-ohlcv` and `/kronos/generate-paths`
- `timestral`: `/timestral/forecast`, `/timestral/forecast-batch`, and `/timestral/forecast-with-covariates`

You can change the forecast length with `--horizon` and, for Kronos, sampled path count with `--num-samples`.

## Notes on runtime support

- `TimesFM` uses the Hugging Face `transformers` runtime by default and can expose additional quantile capability when the official `timesfm` runtime is installed.
- `Chronos-2` is loaded through `chronos-forecasting`.
- `Kronos` is installed into the image from the official source repository during Docker build, then loaded from `KRONOS_RUNTIME_PATH`.
- `Timestral` uses Mistral's standalone `timestral` inference package, installed from its GitHub repository at a pinned commit. It needs no `transformers` runtime and runs in float32 on CPU or GPU.

## Kronos runtime packaging

For `MODEL_TYPE=kronos`, the Docker build now:

1. Clones the official runtime from `https://github.com/shiyu-coder/Kronos.git`
2. Checks out `KRONOS_RUNTIME_REF` (default: `master`)
3. Installs the runtime's `requirements.txt`
4. Copies the runtime source into `/opt/kronos-runtime` in the final image

This removes the previous requirement that the deployment environment manually provide an importable `model.py`.

## Timestral runtime packaging and license

For `MODEL_TYPE=timestral`, the Docker build:

1. Installs the standalone `timestral` package from `https://github.com/mistralai/timestral.git`
2. Checks out `TIMESTRAL_REF` (default: `a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff`)
3. Downloads `mistralai/Timestral-209M` (`config.json`, `model.safetensors`, and the license)
4. Loads the checkpoint offline from `/app/models/timestral`

The `timestral` source code is Apache-2.0. The model weights are released under a Modified MIT license; the downloaded license file is kept next to the weights in the image. Review that license before redistributing the model image.

## Timestral integration notes

- Timestral outputs a fixed grid of 21 non-crossing quantiles. Requests select a subset via `quantiles`; the full grid is listed at `metadata.available_quantiles` in `GET /models/current`.
- `mean` is the 0.5 quantile. Timestral has no separate mean head, so `/predict` uses the median as its point forecast.
- Univariate output is flat (`[horizon]`); multivariate output is nested (`[variates][horizon]`) and includes `variate_count`. The response always includes `variate_count`, so callers can branch without inspecting list depth.
- Covariates are per series: `past_covariates` covers the context, `known_covariates` covers `context + horizon`, and categorical variants accept numeric category codes.
- Series in one request may have different lengths; shorter series are left-padded with missing values. Series with different variate counts are forecast in separate batches.
- Context longer than `TIMESTRAL_MAX_CONTEXT` keeps the most recent steps, matching the upstream runtime.
