"""Integration tests against a real Timestral checkpoint.

These tests are opt-in because the 209M weights are large. Point
`TIMESTRAL_REAL_MODEL` at a local checkpoint directory (for example
`models/timestral` after `scripts/download_models.py --model timestral`):

    TIMESTRAL_REAL_MODEL=$PWD/models/timestral pytest tests/test_timestral_real_model.py

They exercise the same code path as production: the real `timestral` runtime,
the provider mapping, HTTP routes, and MCP tool invocation.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

pytest.importorskip("timestral", reason="Timestral runtime is optional; install it to run these tests.")

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.providers.timestral import TimestralProvider

MODEL_PATH = os.getenv("TIMESTRAL_REAL_MODEL")
pytestmark = pytest.mark.skipif(
    not MODEL_PATH,
    reason="Set TIMESTRAL_REAL_MODEL to a Timestral checkpoint directory to run real-model tests.",
)


@pytest.fixture(scope="module")
def provider() -> TimestralProvider:
    instance = TimestralProvider()
    instance.load(MODEL_PATH, "cpu")
    return instance


def context(length: int, offset: float = 0.0) -> list[float]:
    rng = np.random.default_rng(11)
    return (np.cumsum(rng.standard_normal(length)) + offset).round(4).tolist()


def test_descriptor_matches_209m_checkpoint(provider: TimestralProvider) -> None:
    descriptor = provider.descriptor()
    assert descriptor.metadata["parameter_count"] >= 200_000_000
    assert descriptor.metadata["context_length"] == 32_768
    assert descriptor.metadata["patch_size"] == 32
    assert descriptor.metadata["block_size"] == 32
    assert descriptor.metadata["decode_block_length"] == 1_024
    assert descriptor.metadata["available_quantiles"] == list(provider.quantile_levels)
    assert len(provider.quantile_levels) == 21
    assert descriptor.input_modes == ["univariate", "multivariate", "covariates"]
    assert [task.name for task in descriptor.tasks] == ["forecast_quantile", "forecast_with_covariates"]


def test_all_quantiles_are_monotone(provider: TimestralProvider) -> None:
    values = context(96, 20.0)
    values[0] = None
    values[7] = None
    output = provider.invoke(
        "forecast_quantile",
        {"series": [{"target": values}], "horizon": 7, "quantiles": list(provider.quantile_levels)},
    )
    forecast = output["forecasts"][0]
    assert forecast["variate_count"] == 1
    assert len(forecast["mean"]) == 7
    assert list(forecast["quantiles"]) == [f"{level:g}" for level in provider.quantile_levels]
    for _level, row in forecast["quantiles"].items():
        assert len(row) == 7
        assert all(math.isfinite(value) for value in row)
    keys = [f"{level:g}" for level in provider.quantile_levels]
    for lower, upper in zip(keys, keys[1:], strict=False):
        assert np.all(np.asarray(forecast["quantiles"][lower]) <= np.asarray(forecast["quantiles"][upper]) + 1e-5)
    assert forecast["mean"] == forecast["quantiles"]["0.5"]


def test_multivariate_and_batch_padding_are_neutral(provider: TimestralProvider) -> None:
    long_values = context(80, 100.0)
    short_values = context(48, 50.0)
    batch = provider.invoke(
        "forecast_quantile",
        {
            "series": [
                {"item_id": "long", "target": [long_values]},
                {"item_id": "short", "target": [short_values]},
            ],
            "horizon": 6,
            "quantiles": [0.1, 0.5, 0.9],
        },
    )
    for index, values, name in (
        (0, long_values, "long"),
        (1, short_values, "short"),
    ):
        single = provider.invoke(
            "forecast_quantile",
            {"series": [{"target": [values]}], "horizon": 6, "quantiles": [0.1, 0.5, 0.9]},
        )
        for level in ("0.1", "0.5", "0.9"):
            assert np.allclose(
                batch["forecasts"][index]["quantiles"][level],
                single["forecasts"][0]["quantiles"][level],
                atol=1e-5,
            ), (name, level)

    multi = provider.invoke(
        "forecast_quantile",
        {
            "series": [
                {
                    "target": [long_values, (np.asarray(long_values) * 0.01 + 2).tolist()],
                    "variate_names": ["demand", "price"],
                }
            ],
            "horizon": 5,
            "quantiles": [0.1, 0.5, 0.9],
        },
    )["forecasts"][0]
    assert multi["variate_count"] == 2
    assert len(multi["mean"]) == 2
    assert all(len(row) == 5 for row in multi["quantiles"]["0.5"])


def test_covariates_match_direct_runtime(provider: TimestralProvider) -> None:
    values = context(64, 10.0)
    horizon = 6
    past = np.sin(np.arange(64) / 5).round(4).tolist()
    known = np.sin(np.arange(64 + horizon) / 5).round(4).tolist()
    categorical = [float(2**26 + index % 3) for index in range(64)]

    payload = {
        "series": [
            {
                "target": [values],
                "past_covariates": [past],
                "known_covariates": [known],
                "categorical_past_covariates": [categorical],
            }
        ],
        "horizon": horizon,
        "quantiles": [0.1, 0.5, 0.9],
    }
    output = provider.invoke("forecast_with_covariates", payload)["forecasts"][0]

    direct = provider.forecaster.forecast(
        np.asarray([values], dtype=np.float32)[None, :, :],
        prediction_length=horizon,
        past_covariates=np.asarray([past], dtype=np.float32)[None, :, :],
        known_covariates=np.asarray([known], dtype=np.float32)[None, :, :],
        categorical_past_covariates=np.asarray([categorical], dtype=np.float64)[None, :, :],
    )
    for level in (0.1, 0.5, 0.9):
        assert np.allclose(
            output["quantiles"][f"{level:g}"],
            direct.quantile(level).numpy()[0],
            atol=2e-5,
        ), level


def test_long_horizon_rolls_out_and_clips_nonnegative(provider: TimestralProvider) -> None:
    values = (np.abs(np.asarray(context(64))) + 1).round(4).tolist()
    output = provider.invoke(
        "forecast_quantile",
        {"series": [{"target": values}], "horizon": 1_025, "quantiles": [0.01, 0.5, 0.99]},
    )["forecasts"][0]
    assert len(output["mean"]) == 1_025
    for row in output["quantiles"].values():
        assert len(row) == 1_025
        assert all(math.isfinite(value) for value in row)
        assert min(row) >= 0

    unclipped = provider.invoke(
        "forecast_quantile",
        {
            "series": [{"target": values}],
            "horizon": 8,
            "quantiles": [0.01, 0.5, 0.99],
            "clip_nonnegative": False,
        },
    )["forecasts"][0]
    assert len(unclipped["mean"]) == 8


def test_http_routes_and_mcp(provider: TimestralProvider) -> None:
    values = context(48, 40.0)
    horizon = 4
    app = create_app(settings=Settings(model_type="timestral", api_key="real-key"), provider=provider)
    headers = {"Authorization": "Bearer real-key"}
    mcp_headers = {"Authorization": "Bearer real-key", "Accept": "application/json"}

    with TestClient(app) as client:
        simple = client.post(
            "/timestral/forecast",
            headers=headers,
            json={"series": values, "horizon": horizon, "quantiles": [0.1, 0.5, 0.9]},
        )
        assert simple.status_code == 200, simple.text
        assert set(simple.json()) == {"mean", "quantiles"}

        switched = client.post(
            "/timestral/forecast",
            headers=headers,
            json={
                "series": values,
                "horizon": horizon,
                "quantiles": [0.1, 0.5, 0.9],
                "symmetric_averaging": False,
                "clip_nonnegative": False,
            },
        )
        assert switched.status_code == 200, switched.text
        switched_batch = client.post(
            "/timestral/forecast-batch",
            headers=headers,
            json={
                "series": [{"target": [values]}],
                "horizon": horizon,
                "quantiles": [0.1, 0.5, 0.9],
                "symmetric_averaging": False,
                "clip_nonnegative": False,
            },
        )
        assert switched_batch.status_code == 200, switched_batch.text
        for level in ("0.1", "0.5", "0.9"):
            assert np.allclose(
                switched.json()["quantiles"][level],
                switched_batch.json()["forecasts"][0]["quantiles"][level],
                atol=2e-5,
            ), level

        batch = client.post(
            "/timestral/forecast-batch",
            headers=headers,
            json={
                "series": [{"target": [values, (np.asarray(values) * 0.1).tolist()]}],
                "horizon": horizon,
                "quantiles": [0.1, 0.5, 0.9],
            },
        )
        assert batch.status_code == 200, batch.text
        assert batch.json()["forecasts"][0]["variate_count"] == 2

        covariate = client.post(
            "/timestral/forecast-with-covariates",
            headers=headers,
            json={
                "series": [{"target": values, "known_covariates": [values + [0.0] * horizon]}],
                "horizon": horizon,
                "quantiles": [0.1, 0.5, 0.9],
            },
        )
        assert covariate.status_code == 200, covariate.text

        legacy = client.post(
            "/predict",
            headers=headers,
            json={"instances": [{"history": values}], "task": {"horizon": horizon}},
        )
        assert legacy.status_code == 200, legacy.text
        assert legacy.json()["model"] == "timestral"

        tools = client.post(
            "/mcp/",
            headers=mcp_headers,
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        )
        assert tools.status_code == 200
        assert "invoke_task" in {tool["name"] for tool in tools.json()["result"]["tools"]}

        invoked = client.post(
            "/mcp/",
            headers=mcp_headers,
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "invoke_task",
                    "arguments": {
                        "task": "forecast_quantile",
                        "input": {"series": [{"target": values}], "horizon": horizon},
                    },
                },
            },
        )
        assert invoked.status_code == 200
        assert invoked.json()["result"]["isError"] is False


def test_max_context_override(provider: TimestralProvider) -> None:
    values = context(200, 10.0)
    capped = TimestralProvider(max_context=64)
    capped.load(MODEL_PATH, "cpu")
    try:
        assert capped.forecaster.max_context == 64
        output = capped.invoke(
            "forecast_quantile",
            {"series": [{"target": values}], "horizon": 5, "quantiles": [0.1, 0.5, 0.9]},
        )["forecasts"][0]
        direct = capped.forecaster.forecast(
            np.asarray([values], dtype=np.float32)[None, :, :],
            prediction_length=5,
        )
        assert np.allclose(output["quantiles"]["0.5"], direct.quantile(0.5).numpy()[0], atol=2e-5)
    finally:
        del capped
