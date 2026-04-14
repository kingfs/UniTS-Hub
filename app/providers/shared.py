from __future__ import annotations

from typing import Any, Dict, List


UNIVARIATE_SERIES_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["series", "horizon"],
    "properties": {
        "series": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["target"],
                "properties": {
                    "target": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "number"},
                    },
                    "item_id": {"type": "string"},
                    "frequency": {"type": "string"},
                },
            },
        },
        "horizon": {"type": "integer", "minimum": 1},
        "quantiles": {
            "type": "array",
            "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "frequency": {"type": "string"},
        "num_samples": {"type": "integer", "minimum": 1},
    },
}


POINT_FORECAST_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["forecasts"],
    "properties": {
        "forecasts": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["mean"],
                "properties": {
                    "mean": {"type": "array", "items": {"type": "number"}},
                    "quantiles": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "array",
                            "items": {"type": "number"},
                        },
                    },
                },
            },
        }
    },
}


MULTIVARIATE_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["series", "horizon"],
    "properties": {
        "series": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["target"],
                "properties": {
                    "item_id": {"type": "string"},
                    "timestamps": {"type": "array", "items": {"type": "string"}},
                    "target": {
                        "type": "array",
                        "items": {
                            "oneOf": [
                                {"type": "number"},
                                {
                                    "type": "array",
                                    "items": {"type": "number"},
                                },
                            ]
                        },
                    },
                    "past_covariates": {
                        "type": "array",
                        "items": {"type": "object", "additionalProperties": {"type": "number"}},
                    },
                    "future_covariates": {
                        "type": "array",
                        "items": {"type": "object", "additionalProperties": {"type": "number"}},
                    },
                },
            },
        },
        "horizon": {"type": "integer", "minimum": 1},
        "quantiles": {
            "type": "array",
            "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
    },
}


OHLCV_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["series", "horizon"],
    "properties": {
        "series": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["candles"],
                "properties": {
                    "symbol": {"type": "string"},
                    "candles": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "required": ["open", "high", "low", "close"],
                            "properties": {
                                "timestamp": {"type": "string"},
                                "open": {"type": "number"},
                                "high": {"type": "number"},
                                "low": {"type": "number"},
                                "close": {"type": "number"},
                                "volume": {"type": "number"},
                                "amount": {"type": "number"},
                            },
                        },
                    },
                },
            },
        },
        "horizon": {"type": "integer", "minimum": 1},
        "num_samples": {"type": "integer", "minimum": 1},
    },
}


OHLCV_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["forecasts"],
    "properties": {
        "forecasts": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["candles"],
                "properties": {
                    "symbol": {"type": "string"},
                    "candles": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["open", "high", "low", "close"],
                            "properties": {
                                "timestamp": {"type": "string"},
                                "open": {"type": "number"},
                                "high": {"type": "number"},
                                "low": {"type": "number"},
                                "close": {"type": "number"},
                                "volume": {"type": "number"},
                                "amount": {"type": "number"},
                            },
                        },
                    },
                    "paths": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "object", "additionalProperties": {"type": "number"}},
                        },
                    },
                },
            },
        }
    },
}


def schema_bundle(input_schema: Dict[str, Any], output_schema: Dict[str, Any]) -> Dict[str, Any]:
    return {"input": input_schema, "output": output_schema}


def forecast_result(
    mean: List[float],
    quantiles: Dict[str, List[float]] | None = None,
) -> Dict[str, Any]:
    return {
        "mean": mean,
        "quantiles": quantiles or {},
    }
