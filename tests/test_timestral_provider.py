from __future__ import annotations

import math

import pytest

pytest.importorskip("timestral", reason="Timestral runtime is optional; install it to run these tests.")

import torch
from pydantic import ValidationError
from safetensors.torch import save_file
from timestral import TimestralConfig, TimestralModel

from app.providers.timestral import TimestralProvider

# The real 209M checkpoint uses the same 21-knot grid; a smaller model keeps the
# integration test fast while exercising the real runtime code path end to end.
KNOTS = (
    0.01,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.99,
)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> str:
    directory = tmp_path_factory.mktemp("timestral-checkpoint")
    config = TimestralConfig(
        patch_size=8,
        d_model=32,
        num_heads=4,
        mlp_hidden_dim=64,
        block_schedule=("time", "time", "time_variate"),
        quantile_knots=KNOTS,
        block_size=4,
        context_length=256,
    )
    torch.manual_seed(0)
    model = TimestralModel(config)
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.05)
    config.save(directory / "config.json")
    save_file(model.eval().state_dict(), directory / "model.safetensors")
    return str(directory)


@pytest.fixture(scope="module")
def provider(checkpoint: str) -> TimestralProvider:
    instance = TimestralProvider()
    instance.load(checkpoint, "cpu")
    return instance


def series(length: int, offset: float = 0.0) -> list[float]:
    return [round(10 * math.sin(index / 3) + offset + index * 0.1, 4) for index in range(length)]


def test_descriptor_advertises_real_capabilities(provider: TimestralProvider) -> None:
    descriptor = provider.descriptor()
    assert descriptor.id == "timestral"
    assert [task.name for task in descriptor.tasks] == [
        "forecast_quantile",
        "forecast_with_covariates",
    ]
    assert descriptor.metadata["point_forecast"] == "median"
    assert descriptor.metadata["supports_known_covariates"] is True
    assert descriptor.metadata["available_quantiles"] == list(KNOTS)


def test_univariate_quantile_forecast(provider: TimestralProvider) -> None:
    output = provider.invoke(
        "forecast_quantile",
        {
            "series": [{"item_id": "sensor", "target": series(40)}],
            "horizon": 6,
        },
    )

    forecast = output["forecasts"][0]
    assert forecast["item_id"] == "sensor"
    assert forecast["variate_count"] == 1
    assert len(forecast["mean"]) == 6
    assert list(forecast["quantiles"]) == ["0.1", "0.5", "0.9"]
    for values in forecast["quantiles"].values():
        assert len(values) == 6
    # The median channel is the point forecast.
    assert forecast["quantiles"]["0.5"] == forecast["mean"]


def test_quantiles_are_non_crossing(provider: TimestralProvider) -> None:
    output = provider.invoke(
        "forecast_quantile",
        {
            "series": [{"target": series(60)}],
            "horizon": 8,
            "quantiles": [0.01, 0.1, 0.5, 0.9, 0.99],
        },
    )
    quantiles = output["forecasts"][0]["quantiles"]
    for lower, upper in zip(["0.01", "0.1", "0.5", "0.9"], ["0.1", "0.5", "0.9", "0.99"]):
        assert all(a <= b for a, b in zip(quantiles[lower], quantiles[upper]))


def test_multivariate_forecast_returns_nested_values(provider: TimestralProvider) -> None:
    output = provider.invoke(
        "forecast_quantile",
        {
            "series": [
                {
                    "target": [series(32), series(32, offset=5.0)],
                    "variate_names": ["demand", "price"],
                }
            ],
            "horizon": 5,
            "quantiles": [0.1, 0.9],
        },
    )

    forecast = output["forecasts"][0]
    assert forecast["variate_count"] == 2
    assert forecast["variate_names"] == ["demand", "price"]
    assert len(forecast["mean"]) == 2
    assert all(len(variate) == 5 for variate in forecast["mean"])
    assert len(forecast["quantiles"]["0.1"]) == 2
    assert all(len(variate) == 5 for variate in forecast["quantiles"]["0.9"])


def test_missing_values_are_accepted(provider: TimestralProvider) -> None:
    history = series(36)
    history[4] = None
    history[5] = None
    output = provider.invoke(
        "forecast_quantile",
        {"series": [{"target": history}], "horizon": 4},
    )
    assert len(output["forecasts"][0]["mean"]) == 4


def test_covariates_task_accepts_past_and_known(provider: TimestralProvider) -> None:
    context = series(24)
    horizon = 4
    output = provider.invoke(
        "forecast_with_covariates",
        {
            "series": [
                {
                    "item_id": "promo",
                    "target": [context],
                    "past_covariates": [[0.2] * len(context)],
                    "known_covariates": [[0.5] * (len(context) + horizon)],
                }
            ],
            "horizon": horizon,
        },
    )
    assert len(output["forecasts"][0]["mean"]) == horizon


def test_categorical_covariates_are_accepted(provider: TimestralProvider) -> None:
    context = series(24)
    horizon = 3
    output = provider.invoke(
        "forecast_with_covariates",
        {
            "series": [
                {
                    "target": [context],
                    "categorical_past_covariates": [[0.0, 1.0] * (len(context) // 2)],
                    "categorical_known_covariates": [[1.0] * (len(context) + horizon)],
                }
            ],
            "horizon": horizon,
        },
    )
    assert len(output["forecasts"][0]["mean"]) == horizon


def test_batch_with_different_lengths_is_supported(provider: TimestralProvider) -> None:
    output = provider.invoke(
        "forecast_quantile",
        {
            "series": [
                {"item_id": "short", "target": series(16)},
                {"item_id": "long", "target": series(40)},
            ],
            "horizon": 3,
        },
    )
    assert [forecast["item_id"] for forecast in output["forecasts"]] == ["short", "long"]
    assert all(len(forecast["mean"]) == 3 for forecast in output["forecasts"])


def test_bare_series_and_history_payloads_are_normalized(provider: TimestralProvider) -> None:
    flat_series = provider.invoke(
        "forecast_quantile",
        {"series": series(20), "horizon": 2},
    )
    assert len(flat_series["forecasts"][0]["mean"]) == 2

    history = provider.invoke(
        "forecast_quantile",
        {"history": series(20), "horizon": 2},
    )
    assert len(history["forecasts"][0]["mean"]) == 2


def test_forecast_quantile_rejects_covariates(provider: TimestralProvider) -> None:
    with pytest.raises(ValueError, match="forecast_with_covariates"):
        provider.invoke(
            "forecast_quantile",
            {
                "series": [{"target": [1.0, 2.0, 3.0], "past_covariates": [[1.0, 2.0, 3.0]]}],
                "horizon": 2,
            },
        )


def test_covariate_task_requires_covariates(provider: TimestralProvider) -> None:
    with pytest.raises(ValidationError, match="requires at least one"):
        provider.invoke(
            "forecast_with_covariates",
            {"series": [{"target": [1.0, 2.0, 3.0]}], "horizon": 2},
        )


def test_known_covariates_must_cover_the_horizon(provider: TimestralProvider) -> None:
    with pytest.raises(ValidationError, match="context \\+ horizon"):
        provider.invoke(
            "forecast_with_covariates",
            {
                "series": [
                    {
                        "target": [1.0, 2.0, 3.0],
                        "known_covariates": [[1.0, 2.0, 3.0]],
                    }
                ],
                "horizon": 2,
            },
        )


def test_unavailable_quantile_is_rejected(provider: TimestralProvider) -> None:
    with pytest.raises(ValueError, match="is not available"):
        provider.invoke(
            "forecast_quantile",
            {"series": [{"target": series(20)}], "horizon": 2, "quantiles": [0.42]},
        )


def test_legacy_predict_returns_the_median(provider: TimestralProvider) -> None:
    forecasts = provider.legacy_predict([series(20)], 3, {"frequency": "1d"})
    assert len(forecasts) == 1
    assert len(forecasts[0]["mean"]) == 3


def test_http_routes_with_the_real_runtime(provider: TimestralProvider) -> None:
    from fastapi.testclient import TestClient

    from app.config import Settings
    from app.main import create_app

    app = create_app(
        settings=Settings(model_type="timestral", api_key="test-key"),
        provider=provider,
    )
    headers = {"Authorization": "Bearer test-key"}

    with TestClient(app) as client:
        # The simple route keeps the exact Chronos-2 request/response contract.
        response = client.post(
            "/timestral/forecast",
            headers=headers,
            json={
                "series": series(20),
                "horizon": 4,
                "quantiles": [0.1, 0.5, 0.9],
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert set(body) == {"mean", "quantiles"}
        assert len(body["mean"]) == 4
        assert set(body["quantiles"]) == {"0.1", "0.5", "0.9"}

        # Multivariate and batched requests use the explicit batch route.
        response = client.post(
            "/timestral/forecast-batch",
            headers=headers,
            json={
                "series": [
                    {
                        "item_id": "multi",
                        "target": [series(20), series(20, offset=3.0)],
                        "variate_names": ["a", "b"],
                    }
                ],
                "horizon": 4,
                "quantiles": [0.1, 0.5, 0.9],
            },
        )
        assert response.status_code == 200
        forecast = response.json()["forecasts"][0]
        assert forecast["variate_count"] == 2
        assert len(forecast["quantiles"]["0.9"]) == 2

        response = client.post(
            "/timestral/forecast-with-covariates",
            headers=headers,
            json={
                "series": [
                    {
                        "target": [series(20)],
                        "past_covariates": [[0.2] * 20],
                        "known_covariates": [[0.4] * 24],
                    }
                ],
                "horizon": 4,
            },
        )
        assert response.status_code == 200

        # The dedicated route only serves the active model family.
        response = client.post(
            "/chronos/forecast",
            headers=headers,
            json={"series": [1.0, 2.0, 3.0], "horizon": 2},
        )
        assert response.status_code == 404

        # Provider-level validation problems surface as a 4xx, not a 500.
        response = client.post(
            "/models/current/invoke",
            headers=headers,
            json={
                "task": "forecast_quantile",
                "input": {
                    "series": [{"target": [1.0, 2.0, 3.0]}],
                    "horizon": 2,
                    "quantiles": [0.42],
                },
            },
        )
        assert response.status_code == 400
