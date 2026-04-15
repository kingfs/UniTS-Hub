#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


def build_url(base_url: str, path: str) -> str:
    return urllib.parse.urljoin(f"{base_url.rstrip('/')}/", path.lstrip("/"))


def request_json(
    method: str,
    url: str,
    api_key: str | None,
    payload: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    headers = {
        "Accept": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url=url, method=method, headers=headers, data=body)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status = response.getcode()
            data = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        status = exc.code
        data = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with status={status}: {data}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc.reason}") from exc

    try:
        parsed = json.loads(data)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{method} {url} returned non-JSON response: {data}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{method} {url} returned unexpected JSON payload: {parsed!r}")
    return status, parsed


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def assert_list_length(value: Any, expected: int, message: str) -> None:
    assert_true(isinstance(value, list), f"{message}: expected list, got {type(value).__name__}")
    assert_true(len(value) == expected, f"{message}: expected length={expected}, got length={len(value)}")


def print_json(label: str, payload: Any) -> None:
    print(f"[info] {label}:")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def timesfm_payload(horizon: int) -> dict[str, Any]:
    return {
        "task": "forecast_point",
        "input": {
            "history": [128.4, 129.1, 130.8, 130.4, 131.6, 133.2, 132.8, 134.5],
            "horizon": horizon,
            "frequency": "1d",
        },
    }


def chronos_payload(horizon: int) -> dict[str, Any]:
    return {
        "task": "forecast_quantile",
        "input": {
            "series": [210.0, 212.5, 211.2, 214.8, 217.4, 216.1, 219.3, 221.0],
            "horizon": horizon,
            "quantiles": [0.1, 0.5, 0.9],
        },
    }


def kronos_forecast_payload(horizon: int) -> dict[str, Any]:
    return {
        "task": "forecast_ohlcv",
        "input": {
            "symbol": "AAPL",
            "candles": [
                {
                    "timestamp": "2026-04-01T00:00:00Z",
                    "open": 189.8,
                    "high": 191.1,
                    "low": 188.9,
                    "close": 190.7,
                    "volume": 52340000,
                },
                {
                    "timestamp": "2026-04-02T00:00:00Z",
                    "open": 190.9,
                    "high": 192.2,
                    "low": 190.1,
                    "close": 191.8,
                    "volume": 48720000,
                },
                {
                    "timestamp": "2026-04-03T00:00:00Z",
                    "open": 191.7,
                    "high": 193.0,
                    "low": 190.8,
                    "close": 192.4,
                    "volume": 50150000,
                },
            ],
            "horizon": horizon,
        },
    }


def kronos_paths_payload(horizon: int, num_samples: int) -> dict[str, Any]:
    payload = kronos_forecast_payload(horizon)
    payload["task"] = "generate_paths"
    payload["input"]["num_samples"] = num_samples
    return payload


def verify_common_endpoints(base_url: str, api_key: str | None) -> dict[str, Any]:
    status, health = request_json("GET", build_url(base_url, "/health"), api_key=None)
    assert_true(status == 200, "health endpoint should return 200")
    assert_true(health.get("status") in {"ready", "not_ready"}, "health.status should be ready/not_ready")

    _, model = request_json("GET", build_url(base_url, "/models/current"), api_key=api_key)
    assert_true(isinstance(model.get("id"), str) and model["id"], "current model should expose a non-empty id")

    _, schema = request_json("GET", build_url(base_url, "/models/current/schema"), api_key=api_key)
    assert_true(schema.get("model", {}).get("id") == model["id"], "schema.model.id should match current model id")
    assert_true(isinstance(schema.get("schemas"), dict), "schema.schemas should be a dictionary")
    return model


def verify_timesfm(base_url: str, api_key: str | None, horizon: int) -> None:
    payload = timesfm_payload(horizon)
    print_json("timesfm request", payload)
    _, invoke = request_json("POST", build_url(base_url, "/models/current/invoke"), api_key=api_key, payload=payload)
    assert_true(invoke.get("task") == "forecast_point", "invoke task should be forecast_point")
    assert_list_length(invoke["output"]["forecasts"][0]["mean"], horizon, "timesfm invoke mean")
    print_json("timesfm invoke response", invoke)

    _, direct = request_json(
        "POST",
        build_url(base_url, "/timesfm/forecast"),
        api_key=api_key,
        payload=payload["input"],
    )
    assert_list_length(direct["mean"], horizon, "timesfm direct mean")
    print_json("timesfm direct response", direct)


def verify_chronos(base_url: str, api_key: str | None, horizon: int) -> None:
    payload = chronos_payload(horizon)
    print_json("chronos request", payload)
    _, invoke = request_json("POST", build_url(base_url, "/models/current/invoke"), api_key=api_key, payload=payload)
    forecast = invoke["output"]["forecasts"][0]
    assert_true(invoke.get("task") == "forecast_quantile", "invoke task should be forecast_quantile")
    assert_list_length(forecast["mean"], horizon, "chronos invoke mean")
    for quantile in ("0.1", "0.5", "0.9"):
        assert_list_length(forecast["quantiles"][quantile], horizon, f"chronos quantile {quantile}")
    print_json("chronos invoke response", invoke)

    _, direct = request_json(
        "POST",
        build_url(base_url, "/chronos/forecast"),
        api_key=api_key,
        payload=payload["input"],
    )
    assert_list_length(direct["mean"], horizon, "chronos direct mean")
    for quantile in ("0.1", "0.5", "0.9"):
        assert_list_length(direct["quantiles"][quantile], horizon, f"chronos direct quantile {quantile}")
    print_json("chronos direct response", direct)


def verify_kronos(base_url: str, api_key: str | None, horizon: int, num_samples: int) -> None:
    forecast_payload = kronos_forecast_payload(horizon)
    print_json("kronos forecast request", forecast_payload)
    _, invoke_forecast = request_json(
        "POST",
        build_url(base_url, "/models/current/invoke"),
        api_key=api_key,
        payload=forecast_payload,
    )
    candles = invoke_forecast["output"]["forecasts"][0]["candles"]
    assert_true(invoke_forecast.get("task") == "forecast_ohlcv", "invoke task should be forecast_ohlcv")
    assert_list_length(candles, horizon, "kronos invoke candles")
    print_json("kronos invoke forecast response", invoke_forecast)

    _, direct_forecast = request_json(
        "POST",
        build_url(base_url, "/kronos/forecast-ohlcv"),
        api_key=api_key,
        payload=forecast_payload["input"],
    )
    assert_list_length(direct_forecast["candles"], horizon, "kronos direct candles")
    print_json("kronos direct forecast response", direct_forecast)

    paths_payload = kronos_paths_payload(horizon, num_samples)
    print_json("kronos paths request", paths_payload)
    _, invoke_paths = request_json(
        "POST",
        build_url(base_url, "/models/current/invoke"),
        api_key=api_key,
        payload=paths_payload,
    )
    paths = invoke_paths["output"]["forecasts"][0]["paths"]
    assert_true(invoke_paths.get("task") == "generate_paths", "invoke task should be generate_paths")
    assert_list_length(paths, num_samples, "kronos sampled paths")
    for index, path in enumerate(paths):
        assert_list_length(path, horizon, f"kronos sampled path {index}")
    print_json("kronos invoke paths response", invoke_paths)

    _, direct_paths = request_json(
        "POST",
        build_url(base_url, "/kronos/generate-paths"),
        api_key=api_key,
        payload=paths_payload["input"],
    )
    assert_list_length(direct_paths["paths"], num_samples, "kronos direct paths")
    print_json("kronos direct paths response", direct_paths)


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test a deployed UniTS-Hub API service.")
    parser.add_argument("--base-url", default="http://localhost:8000", help="UniTS-Hub service base URL.")
    parser.add_argument("--api-key", default="unitshub-secret", help="Bearer API key.")
    parser.add_argument("--horizon", type=int, default=3, help="Forecast horizon used in test payloads.")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of sampled paths used for Kronos path-generation checks.",
    )
    args = parser.parse_args()

    try:
        model = verify_common_endpoints(args.base_url, args.api_key)
        model_id = model["id"]
        print(f"[info] detected model={model_id} at {args.base_url}")

        if model_id == "timesfm":
            verify_timesfm(args.base_url, args.api_key, args.horizon)
        elif model_id == "chronos":
            verify_chronos(args.base_url, args.api_key, args.horizon)
        elif model_id == "kronos":
            verify_kronos(args.base_url, args.api_key, args.horizon, args.num_samples)
        else:
            raise RuntimeError(f"Unsupported model id returned by service: {model_id}")
    except Exception as exc:
        print(f"[fail] {exc}", file=sys.stderr)
        return 1

    print("[pass] API smoke test completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
