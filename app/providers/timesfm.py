from __future__ import annotations

from typing import Any, Dict, List

import torch

from app.providers.base import ModelProvider
from app.providers.shared import (
    POINT_FORECAST_OUTPUT_SCHEMA,
    UNIVARIATE_SERIES_SCHEMA,
    forecast_result,
    schema_bundle,
)
from app.schemas import ModelDescriptor, TaskDefinition


FREQ_MAP = {
    "auto": 0,
    "high": 0,
    "1h": 0,
    "h": 0,
    "1d": 0,
    "d": 0,
    "medium": 1,
    "1w": 1,
    "w": 1,
    "1m": 1,
    "m": 1,
    "low": 2,
    "1q": 2,
    "q": 2,
    "1y": 2,
    "y": 2,
    "0": 0,
    "1": 1,
    "2": 2,
}


class TimesFMProvider(ModelProvider):
    def __init__(self) -> None:
        super().__init__()
        self.model = None
        self.runtime = "transformers"

    def load(self, model_path: str, device: str) -> None:
        self.device = device
        try:
            import timesfm

            self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_path)
            self.runtime = "timesfm"
        except ImportError:
            from transformers import TimesFmModelForPrediction

            torch_device = torch.device(device)
            dtype = torch.float32 if torch_device.type == "cpu" else torch.bfloat16
            self.model = TimesFmModelForPrediction.from_pretrained(
                model_path,
                torch_dtype=dtype,
                attn_implementation="sdpa",
            ).to(torch_device)
            self.model.eval()
            self.runtime = "transformers"
        self.loaded = True

    def descriptor(self) -> ModelDescriptor:
        tasks = [
            TaskDefinition(
                name="forecast_point",
                title="Point Forecast",
                description="Point forecast for one or more univariate series.",
                input_schema=UNIVARIATE_SERIES_SCHEMA,
                output_schema=POINT_FORECAST_OUTPUT_SCHEMA,
            )
        ]
        if self.runtime == "timesfm":
            tasks.append(
                TaskDefinition(
                    name="forecast_quantile",
                    title="Quantile Forecast",
                    description="Quantile forecast when the official TimesFM runtime is available.",
                    input_schema=UNIVARIATE_SERIES_SCHEMA,
                    output_schema=POINT_FORECAST_OUTPUT_SCHEMA,
                )
            )
        return ModelDescriptor(
            id="timesfm",
            name="TimesFM 2.5",
            version="2.5",
            description="Google Time Series Foundation Model optimized for univariate forecasting.",
            input_modes=["univariate"],
            output_modes=["point_forecast", "quantile_forecast"] if self.runtime == "timesfm" else ["point_forecast"],
            tasks=tasks,
            metadata={
                "supports_quantiles": self.runtime == "timesfm",
                "supports_covariates": False,
                "supports_multivariate": False,
                "runtime": self.runtime,
            },
        )

    def task_schemas(self) -> Dict[str, Dict[str, Any]]:
        schemas = {
            "forecast_point": schema_bundle(
                UNIVARIATE_SERIES_SCHEMA,
                POINT_FORECAST_OUTPUT_SCHEMA,
            )
        }
        if self.runtime == "timesfm":
            schemas["forecast_quantile"] = schema_bundle(
                UNIVARIATE_SERIES_SCHEMA,
                POINT_FORECAST_OUTPUT_SCHEMA,
            )
        return schemas

    def invoke(self, task: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if task not in self.task_schemas():
            raise ValueError(f"TimesFM does not support task [{task}].")
        if self.model is None:
            raise RuntimeError("TimesFM model not loaded.")

        series = payload.get("series") or []
        horizon = int(payload["horizon"])
        histories = [item["target"] for item in series]

        if self.runtime == "timesfm":
            point_forecast, quantile_forecast = self.model.forecast(
                inputs=histories,
                horizon=horizon,
                return_forecast_on_context=False,
            )
            forecasts = []
            quantiles = payload.get("quantiles") or [0.1, 0.5, 0.9]
            for idx, mean in enumerate(point_forecast):
                quantile_map: Dict[str, List[float]] = {}
                if task == "forecast_quantile":
                    for q_idx, q in enumerate(quantiles):
                        if q_idx < len(quantile_forecast[idx]):
                            quantile_map[str(q)] = quantile_forecast[idx][q_idx][:horizon].tolist()
                forecasts.append(forecast_result(mean[:horizon].tolist(), quantile_map))
            return {"forecasts": forecasts}

        freq_raw = str(payload.get("frequency") or payload.get("freq") or "0").lower()
        freq_idx = FREQ_MAP.get(freq_raw, 0)

        inputs = torch.tensor(histories, dtype=self.model.dtype, device=self.model.device)
        freq = torch.tensor([freq_idx] * len(histories), dtype=torch.long, device=self.model.device)

        with torch.no_grad():
            outputs = self.model(
                past_values=inputs,
                freq=freq,
                return_dict=True,
            )
            mean_predictions = outputs.mean_predictions[:, :horizon]

        forecasts = [
            forecast_result(row.tolist())
            for row in mean_predictions.cpu().to(torch.float32).numpy()
        ]
        return {"forecasts": forecasts}
