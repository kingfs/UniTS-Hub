from __future__ import annotations

import os
from typing import Any, Dict, List

from app.providers.base import ModelProvider
from app.providers.shared import (
    OHLCV_INPUT_SCHEMA,
    OHLCV_OUTPUT_SCHEMA,
    schema_bundle,
)
from app.schemas import ModelDescriptor, TaskDefinition


class KronosProvider(ModelProvider):
    def __init__(self, tokenizer_path: str | None = None) -> None:
        super().__init__()
        self.tokenizer_path = tokenizer_path
        self.predictor = None

    def load(self, model_path: str, device: str) -> None:
        try:
            from model import Kronos, KronosPredictor, KronosTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Kronos runtime is not installed. Install the official Kronos dependencies "
                "from https://github.com/shiyu-coder/Kronos and ensure `model.py` is importable."
            ) from exc

        local_tokenizer = os.path.join(os.path.dirname(model_path), "kronos_tokenizer")
        tokenizer_source = self.tokenizer_path or (
            local_tokenizer if os.path.exists(local_tokenizer) else "NeoQuasar/Kronos-Tokenizer-base"
        )
        tokenizer = KronosTokenizer.from_pretrained(tokenizer_source)
        model = Kronos.from_pretrained(model_path)
        self.predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
        self.loaded = True

    def descriptor(self) -> ModelDescriptor:
        tasks = [
            TaskDefinition(
                name="forecast_ohlcv",
                title="OHLCV Forecast",
                description="Generate future OHLCV candles for one or more assets.",
                input_schema=OHLCV_INPUT_SCHEMA,
                output_schema=OHLCV_OUTPUT_SCHEMA,
            ),
            TaskDefinition(
                name="generate_paths",
                title="Sample Paths",
                description="Generate one or more sampled future candle paths.",
                input_schema=OHLCV_INPUT_SCHEMA,
                output_schema=OHLCV_OUTPUT_SCHEMA,
            ),
        ]
        return ModelDescriptor(
            id="kronos",
            name="Kronos",
            version="1",
            description="Financial foundation model for candlestick forecasting and sampled path generation.",
            input_modes=["ohlcv"],
            output_modes=["ohlcv", "sample_paths"],
            tasks=tasks,
            metadata={
                "supports_sampling": True,
                "recommended_max_context": 512,
            },
        )

    def task_schemas(self) -> Dict[str, Dict[str, Any]]:
        bundle = schema_bundle(OHLCV_INPUT_SCHEMA, OHLCV_OUTPUT_SCHEMA)
        return {
            "forecast_ohlcv": bundle,
            "generate_paths": bundle,
        }

    def default_legacy_task(self) -> str | None:
        return None

    def invoke(self, task: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.predictor is None:
            raise RuntimeError("Kronos model not loaded.")
        if task not in self.task_schemas():
            raise ValueError(f"Kronos does not support task [{task}].")

        import pandas as pd

        horizon = int(payload["horizon"])
        num_samples = int(payload.get("num_samples") or 1)
        temperature = float(payload.get("temperature") or 1.0)
        top_p = float(payload.get("top_p") or 0.9)

        forecasts = []
        for item in payload.get("series") or []:
            candles = item["candles"]
            timestamps = [row.get("timestamp") for row in candles]
            x_df = pd.DataFrame(candles)
            x_timestamp = pd.to_datetime(pd.Series(timestamps))
            y_timestamp = self._future_timestamps(x_timestamp, horizon)
            pred_df = self.predictor.predict(
                df=x_df[[c for c in ["open", "high", "low", "close", "volume", "amount"] if c in x_df.columns]],
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=horizon,
                T=temperature,
                top_p=top_p,
                sample_count=num_samples,
            )
            result = {
                "symbol": item.get("symbol"),
                "candles": self._dataframe_to_candles(pred_df),
            }
            if task == "generate_paths":
                result["paths"] = [result["candles"]]
            forecasts.append(result)
        return {"forecasts": forecasts}

    def _future_timestamps(self, history: Any, horizon: int) -> Any:
        freq = history.diff().dropna().mode()
        step = freq.iloc[0] if not freq.empty else history.iloc[-1] - history.iloc[-2]
        future = [history.iloc[-1] + step * (idx + 1) for idx in range(horizon)]
        import pandas as pd

        return pd.Series(future)

    def _dataframe_to_candles(self, frame: Any) -> List[Dict[str, Any]]:
        reset = frame.reset_index()
        timestamp_key = "index" if "index" in reset.columns else reset.columns[0]
        candles: List[Dict[str, Any]] = []
        for row in reset.to_dict(orient="records"):
            candle = {
                "timestamp": str(row.get(timestamp_key)),
                "open": row.get("open"),
                "high": row.get("high"),
                "low": row.get("low"),
                "close": row.get("close"),
            }
            if "volume" in row:
                candle["volume"] = row.get("volume")
            if "amount" in row:
                candle["amount"] = row.get("amount")
            candles.append(candle)
        return candles
