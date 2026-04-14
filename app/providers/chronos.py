from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from app.providers.base import ModelProvider
from app.providers.shared import (
    MULTIVARIATE_INPUT_SCHEMA,
    POINT_FORECAST_OUTPUT_SCHEMA,
    UNIVARIATE_SERIES_SCHEMA,
    forecast_result,
    schema_bundle,
)
from app.schemas import ModelDescriptor, TaskDefinition


class ChronosProvider(ModelProvider):
    def __init__(self) -> None:
        super().__init__()
        self.pipeline = None
        self.quantiles: List[float] = []

    def load(self, model_path: str, device: str) -> None:
        from chronos import BaseChronosPipeline

        self.device = device
        dtype = torch.bfloat16 if device != "cpu" and torch.cuda.is_available() else torch.float32
        self.pipeline = BaseChronosPipeline.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=dtype,
        )
        self.quantiles = list(getattr(self.pipeline, "quantiles", []))
        self.loaded = True

    def descriptor(self) -> ModelDescriptor:
        tasks = [
            TaskDefinition(
                name="forecast_quantile",
                title="Quantile Forecast",
                description="Quantile forecast for univariate or multivariate inputs.",
                input_schema=MULTIVARIATE_INPUT_SCHEMA,
                output_schema=POINT_FORECAST_OUTPUT_SCHEMA,
            ),
            TaskDefinition(
                name="forecast_multivariate",
                title="Multivariate Forecast",
                description="Forecast multiple target dimensions together.",
                input_schema=MULTIVARIATE_INPUT_SCHEMA,
                output_schema=POINT_FORECAST_OUTPUT_SCHEMA,
            ),
            TaskDefinition(
                name="forecast_with_covariates",
                title="Covariate Forecast",
                description="Forecast with past and future covariates when supported by Chronos-2.",
                input_schema=MULTIVARIATE_INPUT_SCHEMA,
                output_schema=POINT_FORECAST_OUTPUT_SCHEMA,
            ),
        ]
        return ModelDescriptor(
            id="chronos",
            name="Chronos-2",
            version="2",
            description="Amazon Chronos-2 for quantile, multivariate, and covariate-aware forecasting.",
            input_modes=["univariate", "multivariate", "covariates"],
            output_modes=["quantile_forecast"],
            tasks=tasks,
            metadata={
                "default_quantiles": self.quantiles or [0.1, 0.5, 0.9],
            },
        )

    def task_schemas(self) -> Dict[str, Dict[str, Any]]:
        bundle = schema_bundle(MULTIVARIATE_INPUT_SCHEMA, POINT_FORECAST_OUTPUT_SCHEMA)
        return {
            "forecast_quantile": bundle,
            "forecast_multivariate": bundle,
            "forecast_with_covariates": bundle,
            "forecast_point": schema_bundle(UNIVARIATE_SERIES_SCHEMA, POINT_FORECAST_OUTPUT_SCHEMA),
        }

    def default_legacy_task(self) -> str | None:
        return "forecast_point"

    def invoke(self, task: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.pipeline is None:
            raise RuntimeError("Chronos model not loaded.")
        if task not in self.task_schemas():
            raise ValueError(f"Chronos does not support task [{task}].")

        horizon = int(payload["horizon"])
        contexts = self._build_context(payload.get("series") or [])
        quantiles = payload.get("quantiles") or self.quantiles or [0.1, 0.5, 0.9]

        try:
            forecasts = self.pipeline.predict(
                contexts,
                prediction_length=horizon,
            )
            formatted = self._format_quantile_forecasts(forecasts, quantiles)
        except TypeError:
            num_samples = int(payload.get("num_samples") or 20)
            forecasts = self.pipeline.predict(
                contexts,
                prediction_length=horizon,
                num_samples=num_samples,
            )
            formatted = self._format_sample_forecasts(forecasts)

        return {"forecasts": formatted}

    def _build_context(self, series: List[Dict[str, Any]]) -> List[torch.Tensor]:
        contexts: List[torch.Tensor] = []
        for item in series:
            target = item["target"]
            if target and isinstance(target[0], list):
                contexts.append(torch.tensor(target, dtype=torch.float32))
            else:
                contexts.append(torch.tensor(target, dtype=torch.float32))
        return contexts

    def _format_quantile_forecasts(
        self,
        forecasts: Any,
        requested_quantiles: List[float],
    ) -> List[Dict[str, Any]]:
        available = self.quantiles or requested_quantiles
        q_index = {float(q): idx for idx, q in enumerate(available)}
        results: List[Dict[str, Any]] = []

        for forecast in forecasts:
            values = forecast.detach().cpu().numpy() if hasattr(forecast, "detach") else np.asarray(forecast)
            if values.ndim == 3:
                values = values[0]
            median_idx = q_index.get(0.5, len(values) // 2)
            quantile_map: Dict[str, List[float]] = {}
            for q in requested_quantiles:
                idx = q_index.get(float(q))
                if idx is not None and idx < len(values):
                    quantile_map[str(q)] = values[idx].tolist()
            results.append(forecast_result(values[median_idx].tolist(), quantile_map))
        return results

    def _format_sample_forecasts(self, forecasts: Any) -> List[Dict[str, Any]]:
        raw = forecasts.detach().cpu().numpy() if hasattr(forecasts, "detach") else np.asarray(forecasts)
        results: List[Dict[str, Any]] = []
        for sample in raw:
            median = np.quantile(sample, 0.5, axis=0)
            p10 = np.quantile(sample, 0.1, axis=0)
            p90 = np.quantile(sample, 0.9, axis=0)
            results.append(
                forecast_result(
                    median.tolist(),
                    {
                        "0.1": p10.tolist(),
                        "0.5": median.tolist(),
                        "0.9": p90.tolist(),
                    },
                )
            )
        return results
