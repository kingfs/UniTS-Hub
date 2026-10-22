# Timestral Integration

`Timestral` is Mistral's time-series foundation model family (5M, 28M, 123M, 209M). UniTS-Hub serves it through `TimestralProvider` and the standalone `timestral` inference package.

## Why it needs its own provider

Timestral is not a drop-in variant of Chronos or TimesFM:

| Property | Timestral | Consequence for UniTS-Hub |
| --- | --- | --- |
| Multivariate targets | Variates of one series are forecast jointly as `(variates, time)` | `target` is a matrix; a flat array means one variate |
| Covariates | Past-only, known-future, and categorical, per series | A second task, `forecast_with_covariates` |
| Output | Fixed grid of non-crossing quantiles (21 for 209M); no mean, no samples | `mean` is the 0.5 quantile; no `generate_paths` |
| Decoding | Patch transformer with block-autoregressive rollout over 32 patches (1024 steps) | Long horizons are decoded block by block inside the runtime |
| Context | Up to 32768 steps; longer contexts keep the most recent steps | `TIMESTRAL_MAX_CONTEXT` can cap it for latency |
| Missing values | `NaN` is a first-class missing marker with internal forward-fill | JSON `null` maps to `NaN` |
| Runtime | Standalone package; only torch, numpy, safetensors, huggingface_hub | No `transformers` or tokenizer assets |

The 209M checkpoint is a 24-layer causal patch transformer with `patch_size=32`, `d_model=768`, 12 heads, `block_size=32`, `context_length=32768`, and 21 quantile knots (`0.01, 0.05, 0.1 ... 0.9, 0.95, 0.99`). It is trained with Decoupled Horizon Forecasting and supports known-future covariates.

## Uniform usage

The common single-series case is intentionally identical to Chronos-2. Only the model-specific route changes:

| Route | Request | Response |
| --- | --- | --- |
| `POST /chronos/forecast` | `{"series": [...], "horizon": 7, "quantiles": [...]}` | `{"mean": [...], "quantiles": {...}}` |
| `POST /timestral/forecast` | the same payload | the same payload shape |
| `POST /timestral/forecast-batch` | `{"series": [{"target": [[...]]}], ...}` | `{"forecasts": [{"mean": ..., "quantiles": ...}]}` |
| `POST /timestral/forecast-with-covariates` | target + covariates | `{"forecasts": [...]}` |

Both quantile models share `QuantileForecastRequest` / `QuantileForecastResponse` in `app/schemas.py`, so the simple path cannot drift between them. Use the batch route only when you need multiple series, multiple variates, or per-series metadata.

## Task mapping

| UniTS-Hub task | Timestral capability | Notes |
| --- | --- | --- |
| `forecast_quantile` | Target-only probabilistic forecast | Univariate or multivariate; rejects covariate fields |
| `forecast_with_covariates` | Probabilistic forecast conditioned on covariates | Requires at least one covariate field |
| legacy `/predict` | Median point forecast | `default_legacy_task` is `forecast_quantile` |

`generate_paths` is deliberately not advertised: Timestral produces quantiles, not sampled paths.

## Request contract

Canonical single-series request (`/timestral/forecast`), identical to Chronos-2:

```json
{
  "series": [120, 125, 118, 131, 135, 133, 140],
  "horizon": 3,
  "quantiles": [0.1, 0.5, 0.9]
}
```

The response is `{"mean": [...], "quantiles": {"0.1": [...], "0.5": [...], "0.9": [...]}}`.

The rich contract below is used by `/timestral/forecast-batch`, `/timestral/forecast-with-covariates`, and the generic `/models/current/invoke` tasks:

```json
{
  "series": [
    {
      "item_id": "store-42",
      "target": [
        [120, 125, 118, null, 135, 133, 140],
        [9.9, 9.8, 10.1, 10.0, 10.3, null, 10.4]
      ],
      "variate_names": ["demand", "price"],
      "past_covariates": [[0, 1, 0, 0, 1, 1, 0]],
      "known_covariates": [[0, 1, 1, 0, 0, 1, 1, 0, 1, 0]],
      "categorical_known_covariates": [[2, 2, 2, 2, 2, 2, 2, 1, 1, 1]]
    }
  ],
  "horizon": 3,
  "quantiles": [0.1, 0.5, 0.9],
  "symmetric_averaging": true,
  "clip_nonnegative": true
}
```

Rules:

- `target` is `(variates, time)`. A flat `[120, 125, 118]` is accepted as one variate.
- All variates of a series must share the same history length.
- `past_covariates` and `categorical_past_covariates` cover exactly `time` steps.
- `known_covariates` and `categorical_known_covariates` cover `time + horizon` steps; the last `horizon` values are the known future.
- `null` is missing; do not substitute zero.
- `quantiles` must be a subset of the checkpoint's knots. The available grid is published at `metadata.available_quantiles` in `GET /models/current`.
- Aliases `history` (for `target`) and `future_covariates` / `categorical_future_covariates` (for the known variants) are accepted for agent compatibility.

## Response contract

Univariate:

```json
{
  "forecasts": [
    {
      "item_id": "store-42",
      "variate_names": null,
      "variate_count": 1,
      "mean": [133.1, 134.4, 135.2],
      "quantiles": {
        "0.1": [131.8, 132.9, 133.6],
        "0.5": [133.1, 134.4, 135.2],
        "0.9": [134.5, 135.9, 136.8]
      }
    }
  ]
}
```

Multivariate (`variate_count > 1`) uses one nested list per variate:

```json
{
  "forecasts": [
    {
      "item_id": "store-42",
      "variate_names": ["demand", "price"],
      "variate_count": 2,
      "mean": [[133.1, 134.4, 135.2], [10.35, 10.4, 10.45]],
      "quantiles": {
        "0.1": [[131.8, 132.9, 133.6], [10.20, 10.24, 10.28]],
        "0.5": [[133.1, 134.4, 135.2], [10.35, 10.40, 10.45]],
        "0.9": [[134.5, 135.9, 136.8], [10.50, 10.56, 10.62]]
      }
    }
  ]
}
```

`variate_count` lets callers branch without inspecting list depth. The values above are illustrative.

## Batching and shape handling

- Series in one request may have different history lengths. The provider left-pads shorter series with missing values, which Timestral handles natively; results stay right-aligned on the horizon.
- Series with different variate or covariate layouts are split into independent groups, so a request may mix univariate and multivariate series.
- Covariate channel counts must match within a layout group; use separate requests if they differ.
- Output quantiles are non-crossing, and `mean` is exactly the `0.5` series.

## Runtime and performance

- 209M loads in float32 (about 837 MB of weights) and runs on CPU or GPU.
- `symmetric_averaging=true` (the upstream default) performs two forward passes and averages the reflected forecast. Disable it only if you have measured the accuracy tradeoff.
- `clip_nonnegative=true` clamps at zero for series whose observed context is entirely non-negative.
- Horizons longer than one decode block (1024 steps for 209M) roll out block by block, feeding each block's median back as context. Long horizons therefore cost more than one forward pass.
- The full 32768-step context is expensive on CPU. Set `TIMESTRAL_MAX_CONTEXT` (for example `2048`) when latency matters; the runtime keeps the most recent steps.
- Inference is synchronous inside the request handler, like the other providers. Use one worker per GPU and avoid long contexts for high-request-rate deployments.

## Packaging

- The `timestral` package is installed from `https://github.com/mistralai/timestral.git` at a pinned commit (`TIMESTRAL_REF`).
- `scripts/download_models.py --model timestral` downloads `config.json`, `model.safetensors`, and the license from `mistralai/Timestral-209M`.
- The provider loads with `local_files_only=True`, so the image starts without network access.
- The code is Apache-2.0; the weights use a Modified MIT license. Keep the downloaded license file with the weights and review it before redistributing images.

## Verification

```bash
# Provider and API contract tests use a tiny local checkpoint and the real runtime.
uv sync --extra timestral
uv run --extra timestral pytest tests/test_timestral_provider.py tests/test_api_v2.py -q

# Against a running 209M container:
uv run --extra timestral python scripts/api_smoke_test.py \
  --base-url http://localhost:8000 --api-key unitshub-secret
```
