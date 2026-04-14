from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from app.schemas import ModelDescriptor


class ModelProvider(ABC):
    def __init__(self) -> None:
        self.loaded = False
        self.device: str | None = None

    @abstractmethod
    def load(self, model_path: str, device: str) -> None:
        pass

    @abstractmethod
    def descriptor(self) -> ModelDescriptor:
        pass

    @abstractmethod
    def task_schemas(self) -> Dict[str, Dict[str, Any]]:
        pass

    @abstractmethod
    def invoke(self, task: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def default_legacy_task(self) -> str | None:
        return "forecast_point"

    def supports_task(self, task: str) -> bool:
        return task in self.task_schemas()

    def legacy_predict(
        self,
        history: List[List[float]],
        horizon: int,
        parameters: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        task = self.default_legacy_task()
        if not task:
            raise NotImplementedError("Legacy /predict is not supported by this model.")

        output = self.invoke(
            task,
            {
                "series": [{"target": values} for values in history],
                "horizon": horizon,
                **parameters,
            },
        )

        forecasts = output.get("forecasts")
        if not isinstance(forecasts, list):
            raise ValueError("Provider must return a forecasts list for legacy requests.")
        return forecasts
