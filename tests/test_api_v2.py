from __future__ import annotations

from contextlib import contextmanager

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.providers.base import ModelProvider
from app.schemas import ModelDescriptor, TaskDefinition


class FakeProvider(ModelProvider):
    def __init__(self) -> None:
        super().__init__()
        self.loaded = True

    def load(self, model_path: str, device: str) -> None:
        self.loaded = True

    def descriptor(self) -> ModelDescriptor:
        return ModelDescriptor(
            id="fake",
            name="Fake Model",
            version="test",
            description="Provider used by API tests.",
            input_modes=["univariate"],
            output_modes=["point_forecast"],
            tasks=[
                TaskDefinition(
                    name="forecast_point",
                    title="Point Forecast",
                    description="Simple fake forecast.",
                    input_schema={"type": "object"},
                    output_schema={"type": "object"},
                )
            ],
        )

    def task_schemas(self):
        bundle = {
            "input": {"type": "object"},
            "output": {"type": "object"},
        }
        return {
            "forecast_point": bundle,
        }

    def invoke(self, task: str, payload: dict):
        horizon = int(payload["horizon"])
        return {
            "forecasts": [
                {
                    "mean": list(range(horizon)),
                    "quantiles": {},
                }
            ]
        }


@contextmanager
def create_test_client():
    app = create_app(
        settings=Settings(model_type="fake", api_key="test-key"),
        provider=FakeProvider(),
    )
    with TestClient(app) as client:
        yield client


AUTH = {"Authorization": "Bearer test-key"}


def test_current_model_descriptor():
    with create_test_client() as client:
        response = client.get("/models/current", headers=AUTH)
        assert response.status_code == 200
        body = response.json()
        assert body["id"] == "fake"
        assert body["tasks"][0]["name"] == "forecast_point"


def test_invoke_model_v2():
    with create_test_client() as client:
        response = client.post(
            "/models/current/invoke",
            headers=AUTH,
            json={
                "task": "forecast_point",
                "input": {
                    "series": [{"target": [1.0, 2.0, 3.0]}],
                    "horizon": 3,
                },
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["task"] == "forecast_point"
        assert body["output"]["forecasts"][0]["mean"] == [0, 1, 2]


def test_legacy_predict_compatibility():
    with create_test_client() as client:
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
    with create_test_client() as client:
        response = client.post(
            "/mcp",
            headers=AUTH,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "invoke_model",
                    "arguments": {
                        "task": "forecast_point",
                        "input": {
                            "series": [{"target": [1.0, 2.0]}],
                            "horizon": 2,
                        },
                    },
                },
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["result"]["structuredContent"]["forecasts"][0]["mean"] == [0, 1]
