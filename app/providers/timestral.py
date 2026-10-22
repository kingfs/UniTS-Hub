from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from app.providers.base import ModelProvider
from app.schemas import (
    TIMESTRAL_COVARIATE_FIELDS,
    ModelDescriptor,
    TaskDefinition,
    TimestralForecastRequest,
    TimestralForecastResponse,
    TimestralForecastWithCovariatesRequest,
    schema_bundle,
)


def _level_key(level: float) -> str:
    """Stable JSON key for a quantile level (0.01 -> "0.01", 0.5 -> "0.5")."""
    return f"{level:g}"


def _to_matrix(rows: Sequence[Sequence[Any]]) -> np.ndarray:
    """Convert a validated (rows, time) payload to float32, mapping null to NaN."""
    return np.array(
        [[np.nan if value is None else float(value) for value in row] for row in rows],
        dtype=np.float32,
    )


def _stack_covariates(items: Sequence[Any], name: str, width: int) -> np.ndarray | None:
    """Right-align per-series covariate matrices into one (batch, channels, width) array.

    Series in a batch may have different context lengths; Timestral treats NaN as missing,
    so left-padding a shorter series is equivalent to starting it later.
    """
    first = getattr(items[0], name, None)
    if not first:
        return None
    channels = len(first)
    stacked = np.full((len(items), channels, width), np.nan, dtype=np.float32)
    for index, item in enumerate(items):
        matrix = _to_matrix(getattr(item, name) or [])
        stacked[index, :, width - matrix.shape[1] :] = matrix
    return stacked


def _payload_has_covariates(payload: Dict[str, Any]) -> bool:
    """Detect covariate fields before validation, so the error can point at the right task."""
    if any(name in payload for name in TIMESTRAL_COVARIATE_FIELDS):
        return True
    series = payload.get("series")
    if isinstance(series, (list, tuple)):
        return any(
            isinstance(item, dict) and any(name in item for name in TIMESTRAL_COVARIATE_FIELDS) for item in series
        )
    return False


class TimestralProvider(ModelProvider):
    """Provider for the Mistral Timestral family (multivariate quantile forecasting).

    Timestral is quantile-only and exposes a fixed grid of non-crossing quantile knots.
    Its runtime accepts ``(batch, variates, time)`` contexts plus past-only and
    known-future covariates; both are mapped onto the UniTS-Hub task contract here.
    """

    def __init__(self, max_context: int | None = None) -> None:
        super().__init__()
        self.max_context = max_context
        self.forecaster = None
        self.config = None
        self.quantile_levels: List[float] = []
        self.median_index = 0
        self.parameter_count = 0
        self.checkpoint: str | None = None

    def load(self, model_path: str, device: str) -> None:
        # Import lazily so images that serve another model family do not need the
        # optional Timestral runtime installed.
        try:
            from timestral import TimestralForecaster
        except ImportError as exc:
            raise RuntimeError(
                "Timestral runtime is not installed. Install the `timestral` package from "
                "https://github.com/mistralai/timestral.git at the commit pinned by TIMESTRAL_REF."
            ) from exc

        if self.max_context is not None and self.max_context < 1:
            raise ValueError(f"TIMESTRAL_MAX_CONTEXT must be positive, got [{self.max_context}]")

        forecaster = TimestralForecaster.from_pretrained(
            model_path,
            device=device,
            max_context=self.max_context,
            # The image bundles config.json + model.safetensors and must start offline.
            local_files_only=True,
        )
        self.forecaster = forecaster
        self.device = device
        self.config = forecaster.config
        self.quantile_levels = [float(level) for level in forecaster.quantile_levels]
        self.median_index = int(forecaster.median_index)
        self.parameter_count = int(sum(parameter.numel() for parameter in forecaster.model.parameters()))
        self.checkpoint = os.path.basename(os.path.normpath(str(model_path)))
        self.loaded = True

    @property
    def supports_covariates(self) -> bool:
        return bool(self.forecaster.supports_covariates) if self.forecaster is not None else False

    @property
    def supports_known_covariates(self) -> bool:
        return bool(self.forecaster.supports_known_covariates) if self.forecaster is not None else False

    def descriptor(self) -> ModelDescriptor:
        tasks = [
            TaskDefinition(
                name="forecast_quantile",
                title="Quantile Forecast",
                description=("Probabilistic forecast for one or more series with jointly forecast variates."),
                input_schema=TimestralForecastRequest.model_json_schema(),
                output_schema=TimestralForecastResponse.model_json_schema(),
            )
        ]
        if self.supports_covariates:
            tasks.append(
                TaskDefinition(
                    name="forecast_with_covariates",
                    title="Quantile Forecast with Covariates",
                    description=("Probabilistic forecast conditioned on past-only and/or known-future covariates."),
                    input_schema=TimestralForecastWithCovariatesRequest.model_json_schema(),
                    output_schema=TimestralForecastResponse.model_json_schema(),
                )
            )

        metadata: Dict[str, Any] = {
            "runtime": "timestral",
            "checkpoint": self.checkpoint,
            "parameter_count": self.parameter_count,
            "point_forecast": "median",
            "default_quantiles": [0.1, 0.5, 0.9],
            "available_quantiles": self.quantile_levels,
            "supports_missing_values": True,
            "supports_sampling": False,
            "supports_known_covariates": self.supports_known_covariates,
            "supports_categorical_covariates": self.supports_covariates,
        }
        if self.config is not None and self.forecaster is not None:
            metadata.update(
                {
                    "context_length": self.config.context_length,
                    "max_context": self.forecaster.max_context,
                    "patch_size": self.config.patch_size,
                    "block_size": self.config.block_size,
                    "decode_block_length": self.config.block_size * self.config.patch_size,
                }
            )

        return ModelDescriptor(
            id="timestral",
            name="Timestral",
            version="1.0",
            description=(
                "Mistral Timestral: causal patch transformer for multivariate probabilistic "
                "forecasting with past-only and known-future covariates."
            ),
            input_modes=["univariate", "multivariate", "covariates"],
            output_modes=["quantile_forecast"],
            tasks=tasks,
            metadata=metadata,
        )

    def task_schemas(self) -> Dict[str, Dict[str, Any]]:
        schemas = {
            "forecast_quantile": schema_bundle(TimestralForecastRequest, TimestralForecastResponse),
        }
        if self.supports_covariates:
            schemas["forecast_with_covariates"] = schema_bundle(
                TimestralForecastWithCovariatesRequest,
                TimestralForecastResponse,
            )
        return schemas

    def default_legacy_task(self) -> str | None:
        # The legacy contract asks for a point forecast; the median is Timestral's
        # canonical point forecast, so /predict can be served faithfully.
        return "forecast_quantile"

    def invoke(self, task: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.forecaster is None:
            raise RuntimeError("Timestral model not loaded.")
        if task not in self.task_schemas():
            raise ValueError(f"Timestral does not support task [{task}].")
        if task == "forecast_quantile" and _payload_has_covariates(payload):
            raise ValueError("forecast_quantile does not accept covariates; use forecast_with_covariates instead.")

        request_model = (
            TimestralForecastWithCovariatesRequest if task == "forecast_with_covariates" else TimestralForecastRequest
        )
        request = request_model.model_validate(payload)

        if request.has_covariates() and not self.supports_covariates:
            raise ValueError("this checkpoint was not trained with covariates")
        if request.has_known_covariates() and not self.supports_known_covariates:
            raise ValueError("this checkpoint cannot condition on known-future covariates")

        levels, level_indices = self._select_levels(request.quantiles)
        return {"forecasts": self._forecast(request, levels, level_indices)}

    def _select_levels(self, requested: Sequence[float]) -> tuple[List[float], List[int]]:
        levels: List[float] = []
        indices: List[int] = []
        for quantile in requested:
            match = next(
                (
                    index
                    for index, level in enumerate(self.quantile_levels)
                    if math.isclose(level, float(quantile), abs_tol=1e-9)
                ),
                None,
            )
            if match is None:
                raise ValueError(f"Quantile [{quantile}] is not available. Choose from {self.quantile_levels}.")
            if self.quantile_levels[match] not in levels:
                levels.append(self.quantile_levels[match])
                indices.append(match)
        return levels, indices

    def _forecast(
        self,
        request: TimestralForecastRequest,
        levels: Sequence[float],
        level_indices: Sequence[int],
    ) -> List[Dict[str, Any]]:
        matrices = [_to_matrix(item.target) for item in request.series]

        # Batch series with the same variate/covariate layout. Timestral needs a
        # rectangular batch, so groups with different layouts are run separately.
        groups: Dict[tuple, List[int]] = {}
        for index, item in enumerate(request.series):
            signature = (matrices[index].shape[0],) + tuple(
                len(getattr(item, name, None) or []) for name in TIMESTRAL_COVARIATE_FIELDS
            )
            groups.setdefault(signature, []).append(index)

        results: Dict[int, Dict[str, Any]] = {}
        for indices in groups.values():
            group_items = [request.series[index] for index in indices]
            group_matrices = [matrices[index] for index in indices]
            for index, result in zip(
                indices,
                self._forecast_group(request, group_items, group_matrices, levels, level_indices),
                strict=True,
            ):
                results[index] = result

        return [results[index] for index in range(len(request.series))]

    def _forecast_group(
        self,
        request: TimestralForecastRequest,
        items: Sequence[Any],
        matrices: Sequence[np.ndarray],
        levels: Sequence[float],
        level_indices: Sequence[int],
    ) -> List[Dict[str, Any]]:
        batch = len(items)
        variates = matrices[0].shape[0]
        max_context = max(matrix.shape[1] for matrix in matrices)

        # Left-pad shorter series with NaN. Timestral treats NaN as missing, so the
        # padding is neutral and all series stay right-aligned on the same horizon.
        context = np.full((batch, variates, max_context), np.nan, dtype=np.float32)
        for index, matrix in enumerate(matrices):
            context[index, :, max_context - matrix.shape[1] :] = matrix

        past_covariates = _stack_covariates(items, "past_covariates", max_context)
        known_covariates = _stack_covariates(items, "known_covariates", max_context + request.horizon)
        categorical_past = _stack_covariates(items, "categorical_past_covariates", max_context)
        categorical_known = _stack_covariates(items, "categorical_known_covariates", max_context + request.horizon)

        forecast = self.forecaster.forecast(
            context,
            prediction_length=request.horizon,
            past_covariates=past_covariates,
            known_covariates=known_covariates,
            categorical_past_covariates=categorical_past,
            categorical_known_covariates=categorical_known,
            symmetric_averaging=request.symmetric_averaging,
            clip_nonnegative=request.clip_nonnegative,
        )
        quantiles = forecast.quantiles.detach().cpu().to(torch.float32).numpy()

        output: List[Dict[str, Any]] = []
        for offset, item in enumerate(items):
            values = quantiles[offset]  # (variates, horizon, quantiles)
            median = values[:, :, self.median_index]
            if variates == 1:
                mean: Any = median[0].tolist()
                quantile_map: Dict[str, Any] = {
                    _level_key(level): values[0, :, index].tolist()
                    for level, index in zip(levels, level_indices, strict=True)
                }
            else:
                mean = [median[variate].tolist() for variate in range(variates)]
                quantile_map = {
                    _level_key(level): [values[variate, :, index].tolist() for variate in range(variates)]
                    for level, index in zip(levels, level_indices, strict=True)
                }
            output.append(
                {
                    "item_id": item.item_id,
                    "variate_names": item.variate_names,
                    "variate_count": variates,
                    "mean": mean,
                    "quantiles": quantile_map,
                }
            )
        return output
