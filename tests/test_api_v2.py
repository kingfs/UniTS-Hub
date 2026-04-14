from __future__ import annotations

from contextlib import contextmanager

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.providers.base import ModelProvider
from app.schemas import ModelDescriptor, TaskDefinition


class FakeProvider(ModelProvider):
    def __init__(self, model_id: str, tasks: list[TaskDefinition]) -> None:
        super().__init__()
        self.loaded = True
        self.model_id = model_id
        self._tasks = tasks

    def load(self, model_path: str, device: str) -> None:
        self.loaded = True

    def descriptor(self) -> ModelDescriptor:
        return ModelDescriptor(
            id=self.model_id,
            name=f"{self.model_id.title()} Model",
            version="test",
            description="Provider used by API tests.",
            input_modes=["test"],
            output_modes=["test"],
            tasks=self._tasks,
            metadata={},
        )

    def task_schemas(self):
        return {
            task.name: {
                "input": {"type": "object"},
                "output": {"type": "object"},
            }
            for task in self._tasks
        }

    def invoke(self, task: str, payload: dict):
        if self.model_id == "timesfm" and task == "forecast_point":
            horizon = int(payload["horizon"])
            return {"forecasts": [{"mean": list(range(horizon)), "quantiles": {}}]}
        if self.model_id == "chronos" and task == "forecast_quantile":
            horizon = int(payload["horizon"])
            mean = list(range(horizon))
            return {
                "forecasts": [
                    {
                        "mean": mean,
                        "quantiles": {
                            "0.1": [x - 1 for x in mean],
                            "0.5": mean,
                            "0.9": [x + 1 for x in mean],
                        },
                    }
                ]
            }
        if self.model_id == "kronos" and task == "forecast_ohlcv":
            horizon = int(payload["horizon"])
            candles = []
            for idx in range(horizon):
                candles.append(
                    {
                        "timestamp": f"2026-01-0{idx + 1}T00:00:00Z",
                        "open": 10.0 + idx,
                        "high": 11.0 + idx,
                        "low": 9.0 + idx,
                        "close": 10.5 + idx,
                        "volume": 1000.0 + idx,
                    }
                )
            return {"forecasts": [{"symbol": payload.get("symbol"), "candles": candles}]}
        if self.model_id == "kronos" and task == "generate_paths":
            horizon = int(payload["horizon"])
            num_samples = int(payload["num_samples"])
            paths = []
            for sample in range(num_samples):
                candles = []
                for idx in range(horizon):
                    candles.append(
                        {
                            "timestamp": f"2026-02-0{idx + 1}T00:00:00Z",
                            "open": 10.0 + idx + sample,
                            "high": 11.0 + idx + sample,
                            "low": 9.0 + idx + sample,
                            "close": 10.5 + idx + sample,
                            "volume": 1000.0 + idx + sample,
                        }
                    )
                paths.append(candles)
            return {"forecasts": [{"symbol": payload.get("symbol"), "paths": paths}]}
        raise ValueError(f"Unsupported fake task [{self.model_id}:{task}]")


def make_tasks(*names: str) -> list[TaskDefinition]:
    return [
        TaskDefinition(
            name=name,
            title=name,
            description="Test task",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )
        for name in names
    ]


@contextmanager
def create_test_client(model_id: str, tasks: list[str]):
    app = create_app(
        settings=Settings(model_type=model_id, api_key="test-key"),
        provider=FakeProvider(model_id=model_id, tasks=make_tasks(*tasks)),
    )
    with TestClient(app) as client:
        yield client


AUTH = {"Authorization": "Bearer test-key"}


def test_current_model_descriptor():
    with create_test_client("timesfm", ["forecast_point"]) as client:
        response = client.get("/models/current", headers=AUTH)
        assert response.status_code == 200
        body = response.json()
        assert body["id"] == "timesfm"
        assert [task["name"] for task in body["tasks"]] == ["forecast_point"]


def test_invoke_model_v2():
    with create_test_client("timesfm", ["forecast_point"]) as client:
        response = client.post(
            "/models/current/invoke",
            headers=AUTH,
            json={
                "task": "forecast_point",
                "input": {
                    "history": [1.0, 2.0, 3.0],
                    "horizon": 3,
                    "frequency": "auto",
                },
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["task"] == "forecast_point"
        assert body["output"]["forecasts"][0]["mean"] == [0, 1, 2]


def test_timesfm_model_route():
    with create_test_client("timesfm", ["forecast_point"]) as client:
        response = client.post(
            "/timesfm/forecast",
            headers=AUTH,
            json={
                "history": [1.0, 2.0, 3.0],
                "horizon": 2,
                "frequency": "1d",
            },
        )
        assert response.status_code == 200
        assert response.json()["mean"] == [0, 1]


def test_chronos_model_route():
    with create_test_client("chronos", ["forecast_quantile"]) as client:
        response = client.post(
            "/chronos/forecast",
            headers=AUTH,
            json={
                "series": [1.0, 2.0, 3.0],
                "horizon": 2,
                "quantiles": [0.1, 0.5, 0.9],
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["mean"] == [0, 1]
        assert body["quantiles"]["0.9"] == [1, 2]


def test_kronos_model_route():
    with create_test_client("kronos", ["forecast_ohlcv", "generate_paths"]) as client:
        response = client.post(
            "/kronos/forecast-ohlcv",
            headers=AUTH,
            json={
                "symbol": "AAPL",
                "candles": [
                    {
                        "timestamp": "2026-01-01T00:00:00Z",
                        "open": 10.0,
                        "high": 11.0,
                        "low": 9.0,
                        "close": 10.5,
                        "volume": 1000.0,
                    }
                ],
                "horizon": 2,
                "temperature": 1.0,
                "top_p": 0.9,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["symbol"] == "AAPL"
        assert len(body["candles"]) == 2


def test_kronos_generate_paths_route():
    with create_test_client("kronos", ["forecast_ohlcv", "generate_paths"]) as client:
        response = client.post(
            "/kronos/generate-paths",
            headers=AUTH,
            json={
                "symbol": "AAPL",
                "candles": [
                    {
                        "timestamp": "2026-01-01T00:00:00Z",
                        "open": 10.0,
                        "high": 11.0,
                        "low": 9.0,
                        "close": 10.5,
                        "volume": 1000.0,
                    }
                ],
                "horizon": 2,
                "num_samples": 3,
                "temperature": 1.0,
                "top_p": 0.9,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["symbol"] == "AAPL"
        assert len(body["paths"]) == 3
        assert len(body["paths"][0]) == 2


def test_model_route_isolated_by_active_model():
    with create_test_client("chronos", ["forecast_quantile"]) as client:
        response = client.post(
            "/timesfm/forecast",
            headers=AUTH,
            json={
                "history": [1.0, 2.0, 3.0],
                "horizon": 2,
                "frequency": "1d",
            },
        )
        assert response.status_code == 404


def test_legacy_predict_compatibility():
    with create_test_client("timesfm", ["forecast_point"]) as client:
        response = client.post(
            "/predict",
            headers=AUTH,
            json={
                "instances": [{"history": [1.0, 2.0, 3.0]}],
                "task": {"horizon": 2},
                "parameters": {},
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["metadata"]["deprecated"] is True
        assert body["forecasts"][0]["mean"] == [0, 1]


def test_mcp_tools_call():
    with create_test_client("timesfm", ["forecast_point"]) as client:
        response = client.post(
            "/mcp",
            headers=AUTH,
            json={
                "jsonrpc": "2.0",
                "id": 1,
            "method": "tools/call",
            "params": {
                    "name": "invoke_task",
                    "arguments": {
                        "task": "forecast_point",
                        "input": {
                            "history": [1.0, 2.0],
                            "horizon": 2,
                            "frequency": "auto",
                        },
                    },
                },
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["result"]["structuredContent"]["forecasts"][0]["mean"] == [0, 1]


def test_mcp_get_task_schema():
    with create_test_client("chronos", ["forecast_quantile"]) as client:
        response = client.post(
            "/mcp",
            headers=AUTH,
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "get_task_schema",
                    "arguments": {
                        "task": "forecast_quantile",
                    },
                },
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert "input" in body["result"]["structuredContent"]
        assert "output" in body["result"]["structuredContent"]
