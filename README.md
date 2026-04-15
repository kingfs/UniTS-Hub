# UniTS-Hub

Capability-first serving for time-series foundation models.

UniTS-Hub v2 keeps the original single-model deployment model, but replaces the old predict-only interface with a model-capability API designed for AI agents. The service now targets three model families:

- `TimesFM 2.5` for univariate point forecasting
- `Chronos-2` for quantile, multivariate, and covariate-oriented forecasting
- `Kronos` for financial OHLCV forecasting and sampled path generation

It exposes three integration surfaces:

- REST API for Dify, LangChain, Flowise, and generic HTTP clients
- MCP endpoint powered by the official [`modelcontextprotocol/python-sdk`](https://github.com/modelcontextprotocol/python-sdk)
- A repo-local Codex skill at [`.agents/skills/unitshub-agent/SKILL.md`](/Users/kingfs/go/src/github.com/kingfs/UniTS-Hub/.agents/skills/unitshub-agent/SKILL.md)

## Core design

- One container still serves one loaded model at a time.
- Agents discover model capabilities at runtime instead of relying on static docs.
- Task schemas are model-aware and exposed explicitly.
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
| `MODEL_TYPE` | `timesfm`, `chronos`, or `kronos` | `chronos` |
| `MODELS_DIR` | Base directory containing model weights | `/app/models` |
| `API_KEY` | Bearer token used by API and MCP | `unitshub-secret` |
| `KRONOS_TOKENIZER_PATH` | Optional local tokenizer path for Kronos | unset |
| `KRONOS_RUNTIME_PATH` | Location of the official Kronos source runtime inside the container | `/opt/kronos-runtime` |

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

## Local development

```bash
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Deployed API smoke test

After you start one model container, you can validate the live service with:

```bash
python3 scripts/api_smoke_test.py --base-url http://localhost:8000 --api-key unitshub-secret
```

The script first calls `/models/current`, detects whether the service is running `timesfm`, `chronos`, or `kronos`, then sends model-specific sample payloads to `/models/current/invoke` and the direct model route:

- `timesfm`: `/timesfm/forecast`
- `chronos`: `/chronos/forecast`
- `kronos`: `/kronos/forecast-ohlcv` and `/kronos/generate-paths`

You can change the forecast length with `--horizon` and, for Kronos, sampled path count with `--num-samples`.

## Notes on runtime support

- `TimesFM` uses the Hugging Face `transformers` runtime by default and can expose additional quantile capability when the official `timesfm` runtime is installed.
- `Chronos-2` is loaded through `chronos-forecasting`.
- `Kronos` is installed into the image from the official source repository during Docker build, then loaded from `KRONOS_RUNTIME_PATH`.

## Kronos runtime packaging

For `MODEL_TYPE=kronos`, the Docker build now:

1. Clones the official runtime from `https://github.com/shiyu-coder/Kronos.git`
2. Checks out `KRONOS_RUNTIME_REF` (default: `master`)
3. Installs the runtime's `requirements.txt`
4. Copies the runtime source into `/opt/kronos-runtime` in the final image

This removes the previous requirement that the deployment environment manually provide an importable `model.py`.
