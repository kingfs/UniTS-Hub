# UniTS-Hub v2 Architecture

## 1. Scope

This document defines the target architecture for UniTS-Hub v2.

The goal is narrow and operational:

- Build Docker images with model weights baked in during GitHub Actions
- Start exactly one model per container at runtime
- Expose inference interfaces that match the real capabilities of that model
- Support both REST and MCP for AI workflow tools and AI agents
- Prefer correctness and clarity over abstract generality

UniTS-Hub v2 is not intended to be a generic model platform. It is a small serving runtime for four specific model families:

- `TimesFM 2.5` / `3.0`
- `Chronos-2`
- `Kronos`
- `Timestral` (Mistral)

## 2. Product Constraints

### 2.1 Single-model runtime

One container loads one model family.

Examples:

- `kingfs/unitshub:timesfm-latest`
- `kingfs/unitshub:chronos2-latest`
- `kingfs/unitshub:kronos-latest`
- `kingfs/unitshub:timestral-latest`

The service may expose a common discovery layer, but runtime behavior is always based on the one active model.

### 2.2 We only expose real capabilities

A capability must not appear in REST, MCP, OpenAPI, or skills unless it is implemented and tested for that model/runtime combination.

This rule is more important than interface uniformity.

### 2.3 Image is self-contained

The published image must contain:

- application code
- Python dependencies
- model weights
- tokenizer assets
- model-specific runtime code required for inference

Runtime should not need to download anything from the network.

## 3. Non-goals

- No multi-model hot-switching in one running container
- No plugin marketplace or arbitrary third-party model loading
- No broad task taxonomy beyond what the four target models actually need
- No capability claims based only on papers or READMEs

## 4. Model Capability Matrix

This table is the contract baseline. Only rows marked `supported` should surface in public interfaces.

| Model | Input shape | Core tasks | Output shape | Status |
| --- | --- | --- | --- | --- |
| TimesFM 2.5 | univariate numeric history + horizon + optional frequency bucket | `forecast_point` | point forecast | required |
| TimesFM 2.5 | univariate numeric history + horizon + optional quantiles | `forecast_quantile` | quantile forecast | optional, only if official runtime is bundled and tested |
| Chronos-2 | univariate or multivariate history | `forecast_quantile` | quantile forecast | required |
| Chronos-2 | history + covariates | `forecast_with_covariates` | quantile forecast | optional, only after dataframe/covariate path is implemented and tested |
| Kronos | OHLC or OHLCV candle history | `forecast_ohlcv` | future candles | required |
| Kronos | OHLC or OHLCV candle history + sample count | `generate_paths` | sampled future paths | required |
| Timestral | univariate or multivariate numeric history + horizon | `forecast_quantile` | quantile forecast on the checkpoint's fixed knot grid | required |
| Timestral | history + past-only and/or known-future covariates | `forecast_with_covariates` | quantile forecast | required when the checkpoint was trained with covariates |

### 4.1 TimesFM

Target contract:

- minimum public task: `forecast_point`
- optional task: `forecast_quantile`
- input is compact and univariate
- no covariates in v2 public contract
- no multivariate support in v2 public contract

### 4.2 Chronos-2

Target contract:

- required task: `forecast_quantile`
- optional task: `forecast_with_covariates`
- multivariate support is exposed only if the actual runtime path is confirmed and tested
- point output is derived from median/0.5 quantile when compatibility endpoints need it

### 4.3 Kronos

Target contract:

- required task: `forecast_ohlcv`
- required task: `generate_paths`
- input is financial candle data
- output is future candles, optionally with multiple sampled paths
- timestamps should be accepted and preserved when available

### 4.4 Timestral

Target contract:

- required task: `forecast_quantile`
- required task: `forecast_with_covariates` when the checkpoint has `covariate_enabled`
- input target is a `(variates, time)` matrix; a flat array is a one-variate series
- variates of one series are forecast jointly, not independently
- covariates are conditioning rows: past-only (`time`) and known-future (`time + horizon`), numeric or categorical
- output is quantile-only on the checkpoint's fixed, non-crossing knot grid (21 levels for the 209M checkpoint)
- `mean` is the 0.5 quantile, because Timestral has no mean head
- `null` marks missing observations; the runtime forward-fills internally
- horizon is decoded block by block beyond one decode block (`block_size * patch_size`, 1024 steps for 209M)
- one request may batch series with different context lengths (left-padded with missing) but not different variate layouts

Capability notes:

- `forecast_quantile` must reject covariate fields so the two tasks stay honest and discoverable.
- `forecast_with_covariates` must reject requests without any covariate field.
- Quantile requests must be a subset of the checkpoint's knots; do not interpolate unverified levels.
- Known-future support depends on the checkpoint's layer schedule and must be read from the runtime, not assumed.
- The runtime has no sampling path, so `generate_paths` must not be advertised.

## 5. Architecture

```text
                +-----------------------------+
                |        UniTS-Hub App        |
                |      FastAPI + MCP HTTP     |
                +--------------+--------------+
                               |
                     current active provider
                               |
        +----------------------+----------------------+----------------------+
        |                      |                      |                      |
   TimesFMProvider      Chronos2Provider       KronosProvider      TimestralProvider
        |                      |                      |                      |
   model runtime          model runtime          model runtime          model runtime
```

### 5.1 Components

- `app/main.py`
  - FastAPI app
  - auth
  - health
  - REST routes
  - MCP route
- `app/providers/*`
  - one provider per model family
  - capability descriptor
  - load model
  - invoke supported tasks
- `app/schemas.py`
  - shared metadata models
  - per-task request and response models
- `scripts/download_models.py`
  - offline asset preparation for image builds

### 5.2 Provider contract

Each provider must implement:

- `load(model_path, device)`
- `descriptor()`
- `task_schemas()`
- `invoke(task, payload)`

Each provider must also define:

- exact supported tasks
- exact runtime requirements
- exact request and response shape

## 6. API Design

The API surface has two layers:

- common discovery layer
- model-specific task layer

This avoids forcing every caller through a single generic abstraction while still allowing agents to discover capabilities.

### 6.1 Common discovery endpoints

These endpoints exist for every image:

- `GET /health`
- `GET /models/current`
- `GET /models/current/schema`
- `GET /models/current/tasks/{task}/schema`
- `POST /models/current/invoke`

#### `GET /models/current`

Purpose:

- identify which model is active
- list supported tasks
- communicate runtime metadata

Example shape:

```json
{
  "id": "chronos",
  "name": "Chronos-2",
  "version": "2",
  "tasks": [
    {
      "name": "forecast_quantile",
      "title": "Quantile Forecast"
    }
  ],
  "metadata": {
    "runtime": "chronos-forecasting"
  }
}
```

#### `POST /models/current/invoke`

Purpose:

- generic agent-friendly invocation path

Request shape:

```json
{
  "task": "forecast_point",
  "input": { "...": "task-specific payload" }
}
```

### 6.2 Model-specific REST endpoints

These endpoints are the preferred public API for human callers and workflow tools.

#### TimesFM

- `POST /timesfm/forecast`

Request:

```json
{
  "history": [1.0, 2.0, 3.0, 4.0],
  "horizon": 8,
  "frequency": "1d"
}
```

Response:

```json
{
  "mean": [4.2, 4.3, 4.4]
}
```

If quantile runtime is enabled:

- `POST /timesfm/forecast-quantile`

#### Chronos-2

- `POST /chronos/forecast`

Baseline request:

```json
{
  "series": [1.0, 2.0, 3.0, 4.0],
  "horizon": 8,
  "quantiles": [0.1, 0.5, 0.9]
}
```

Optional future endpoint after implementation:

- `POST /chronos/forecast-with-covariates`

#### Timestral

- `POST /timestral/forecast` — single univariate series, same contract as `/chronos/forecast`
- `POST /timestral/forecast-batch` — one or more series, optionally multivariate
- `POST /timestral/forecast-with-covariates` — target plus past/known/categorical covariates

The simple route uses the shared `QuantileForecastRequest` / `QuantileForecastResponse`
schema, so Chronos-2 and Timestral are interchangeable for the common case:

```json
{
  "series": [120, 125, 118, 131, 135],
  "horizon": 3,
  "quantiles": [0.1, 0.5, 0.9]
}
```

The batch and covariate routes use the rich request:

```json
{
  "series": [
    {
      "item_id": "store-42",
      "target": [
        [120, 125, 118, 131, 135],
        [9.9, 9.8, 10.1, 10.0, 10.3]
      ],
      "variate_names": ["demand", "price"],
      "past_covariates": [[0, 1, 0, 0, 1]],
      "known_covariates": [[0, 1, 1, 0, 0, 1, 0, 1]]
    }
  ],
  "horizon": 3,
  "quantiles": [0.1, 0.5, 0.9]
}
```

Response:

```json
{
  "forecasts": [
    {
      "item_id": "store-42",
      "variate_names": ["demand", "price"],
      "variate_count": 2,
      "mean": [[133.1, 134.4, 135.2], [10.35, 10.4, 10.45]],
      "quantiles": {"0.1": [[...], [...]], "0.5": [[...], [...]], "0.9": [[...], [...]]}
    }
  ]
}
```

The same payload shape is used by `/models/current/invoke` with task `forecast_quantile` or `forecast_with_covariates`.

#### Kronos

- `POST /kronos/forecast-ohlcv`
- `POST /kronos/generate-paths`

Request:

```json
{
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
  "num_samples": 8
}
```

### 6.3 Compatibility endpoints

Retain for migration only:

- `POST /predict`
- `POST /predict/csv`

Rules:

- only available when the active model can faithfully map to the legacy contract
- must be documented as compatibility-only
- should not be the recommended path in README or skill guidance

## 7. MCP Design

The MCP layer should stay minimal.

### 7.1 Required tools

- `get_current_model`
- `get_task_schema`
- `invoke_task`

### 7.2 Optional tools

- `list_tasks`

### 7.3 Principles

- MCP should reflect the same task set as REST
- no hidden capabilities in MCP
- no broad resource system unless an agent integration actually needs it
- stateless HTTP transport is acceptable for v2 if it is documented clearly

## 8. Schema Strategy

Use strong Pydantic request/response models for every public task.

Do not rely on large `Dict[str, Any]` payloads for public interfaces.

Required schema families:

- `TimesFMForecastRequest`
- `TimesFMForecastResponse`
- `ChronosForecastRequest`
- `ChronosForecastResponse`
- `KronosForecastRequest`
- `KronosForecastResponse`
- `KronosGeneratePathsRequest`
- `KronosGeneratePathsResponse`
- `TimestralSeries`
- `TimestralForecastRequest`
- `TimestralForecastWithCovariatesRequest`
- `TimestralForecastResponse`

These models should drive:

- FastAPI request validation
- OpenAPI generation
- MCP input schema generation
- compatibility adapters where needed

## 9. Docker and Build Pipeline

## 9.1 Image strategy

Preferred strategy:

- build one image per model family

Tags:

- `kingfs/unitshub:timesfm-latest`
- `kingfs/unitshub:chronos2-latest`
- `kingfs/unitshub:kronos-latest`
- `kingfs/unitshub:timestral-latest`
- optional unified semantic tags such as `:timesfm-v2.0.0`

Reason:

- smaller runtime surface
- clearer public contract
- no ambiguity about which model assets are bundled
- easier debugging and release rollback

### 9.2 Docker build args

Required build args:

- `MODEL_TYPE`
- optional `MODEL_REPO`
- optional `TOKENIZER_REPO`

Build behavior:

1. install dependencies for the target model
2. download target model assets only
3. copy only required runtime files into final image
4. set `MODEL_TYPE` default in image metadata or environment

### 9.3 Runtime requirements by image

#### TimesFM image

- include `transformers`
- optionally include official `timesfm` runtime if quantiles are part of the contract

#### Chronos-2 image

- include `chronos-forecasting`
- include any dataframe/covariate dependencies only if covariate support is enabled

#### Kronos image

- include official Kronos runtime code
- include tokenizer assets
- ensure no runtime network dependency remains

#### Timestral image

- install the standalone `timestral` package from its pinned Git commit
- include `config.json`, `model.safetensors`, and the weight license
- no `transformers` runtime or tokenizer assets required
- load from the local checkpoint directory with `local_files_only=True`

## 10. GitHub Actions Release Pipeline

The workflow should become a model matrix build.

### 10.1 Test job

- run unit tests
- run provider smoke tests for request/response transformations
- skip weight-loading tests when weights are absent

### 10.2 Build matrix

Matrix dimensions:

- `model_type`: `timesfm`, `chronos2`, `kronos`, `timestral`
- `platform`: `linux/amd64`, `linux/arm64`

Each matrix job:

1. build model-specific image
2. push digest
3. publish model-specific tags

### 10.3 Manifest publish

Create per-model multi-arch manifests instead of one generic `latest` only.

## 11. Skills

The skill should focus on practical model selection and request shaping.

It should teach agents:

- which model to choose
- which endpoint to call
- what the required fields are
- how to interpret the response

It should not try to replicate all product documentation.

## 12. Implementation Plan

### Phase 1: contract correction

- reduce public task list to only true capabilities
- remove unimplemented Chronos covariate/multivariate claims
- implement real Kronos path sampling or hide the task until done
- decide whether TimesFM quantile is part of the supported contract

### Phase 2: typed schemas and model routes

- add per-task Pydantic models
- add model-specific REST routes
- keep `/models/current/invoke` as generic path
- keep legacy `/predict` as compatibility-only

### Phase 3: packaging

- refactor Dockerfile for model-specific build
- refactor GitHub Actions into model matrix builds
- ensure offline startup for all model images

### Phase 4: MCP tightening

- reduce MCP surface to the minimal stable tool set
- align MCP schema directly with typed request/response models

## 13. Gap Analysis Against Current Code

The current branch still has several mismatches with this target:

1. Chronos advertises tasks that are not yet wired to distinct runtime paths.
2. Kronos `generate_paths` is not yet guaranteed to return real sampled paths.
3. Public routes are still too centered on generic invocation and not enough on model-specific endpoints.
4. Provider payloads still rely heavily on generic dictionaries.
5. Docker and GitHub Actions still build one generic image instead of one image per model family.
6. Kronos runtime packaging is not yet guaranteed to be self-contained.

## 14. Acceptance Criteria

UniTS-Hub v2 is complete when all of the following are true:

- each published image starts offline and loads its bundled model successfully
- every advertised task has an automated test
- every model has at least one model-specific REST endpoint
- `GET /models/current` and MCP report only the active model's real capabilities
- legacy endpoints remain functional where meaningful
- README and skill guidance match the shipped behavior exactly
