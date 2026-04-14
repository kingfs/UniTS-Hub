from __future__ import annotations

from typing import Any, Dict, List

import torch

from app.providers.base import ModelProvider
from app.providers.shared import forecast_result
from app.schemas import (
    ModelDescriptor,
    TaskDefinition,
    TimesFMForecastRequest,
    TimesFMForecastResponse,
    schema_bundle,
)


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
        from transformers import TimesFmModelForPrediction

        torch_device = torch.device(device)
        dtype = torch.float32 if torch_device.type == "cpu" else torch.bfloat16
        self.model = TimesFmModelForPrediction.from_pretrained(
            model_path,
            torch_dtype=dtype,
            attn_implementation="sdpa",
        ).to(torch_device)
        self.model.eval()
        self.loaded = True

    def descriptor(self) -> ModelDescriptor:
        tasks = [
            TaskDefinition(
                name="forecast_point",
                title="Point Forecast",
                description="Point forecast for a univariate time series.",
                input_schema=TimesFMForecastRequest.model_json_schema(),
                output_schema=TimesFMForecastResponse.model_json_schema(),
            )
        ]
        return ModelDescriptor(
            id="timesfm",
            name="TimesFM 2.5",
            version="2.5",
            description="Google Time Series Foundation Model optimized for univariate forecasting.",
            input_modes=["univariate"],
            output_modes=["point_forecast"],
            tasks=tasks,
            metadata={
                "supports_quantiles": False,
                "supports_covariates": False,
                "supports_multivariate": False,
                "runtime": self.runtime,
            },
        )

    def task_schemas(self) -> Dict[str, Dict[str, Any]]:
        return {
            "forecast_point": schema_bundle(TimesFMForecastRequest, TimesFMForecastResponse),
        }

    def invoke(self, task: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if task != "forecast_point":
            raise ValueError(f"TimesFM does not support task [{task}].")
        if self.model is None:
            raise RuntimeError("TimesFM model not loaded.")

        if "series" in payload:
            histories = [item["target"] for item in (payload.get("series") or [])]
            horizon = int(payload["horizon"])
            freq_raw = str(payload.get("frequency") or payload.get("freq") or "0").lower()
        else:
            request = TimesFMForecastRequest.model_validate(payload)
            histories = [request.history]
            horizon = request.horizon
            freq_raw = request.frequency.lower()

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
