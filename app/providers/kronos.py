from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

from app.providers.base import ModelProvider
from app.schemas import (
    KronosForecastRequest,
    KronosForecastResponse,
    KronosGeneratePathsRequest,
    KronosGeneratePathsResponse,
    ModelDescriptor,
    TaskDefinition,
    schema_bundle,
)


class KronosProvider(ModelProvider):
    def __init__(
        self,
        tokenizer_path: str | None = None,
        runtime_path: str | None = None,
    ) -> None:
        super().__init__()
        self.tokenizer_path = tokenizer_path
        self.runtime_path = runtime_path or "/opt/kronos-runtime"
        self.predictor = None

    def load(self, model_path: str, device: str) -> None:
        runtime_root = self.runtime_path
        if runtime_root and runtime_root not in sys.path and os.path.isdir(runtime_root):
            sys.path.insert(0, runtime_root)
        try:
            from model import Kronos, KronosPredictor, KronosTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Kronos runtime is not installed. Build the image with the official Kronos source "
                f"available under KRONOS_RUNTIME_PATH=[{self.runtime_path}] and ensure `model.py` is importable."
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
                description="Generate future OHLC/OHLCV candles for one asset series.",
                input_schema=KronosForecastRequest.model_json_schema(),
                output_schema=KronosForecastResponse.model_json_schema(),
            ),
            TaskDefinition(
                name="generate_paths",
                title="Generate Paths",
                description="Generate multiple sampled future OHLC/OHLCV paths for one asset series.",
                input_schema=KronosGeneratePathsRequest.model_json_schema(),
                output_schema=KronosGeneratePathsResponse.model_json_schema(),
            )
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
                "runtime_path": self.runtime_path,
            },
        )

    def task_schemas(self) -> Dict[str, Dict[str, Any]]:
        return {
            "forecast_ohlcv": schema_bundle(KronosForecastRequest, KronosForecastResponse),
            "generate_paths": schema_bundle(KronosGeneratePathsRequest, KronosGeneratePathsResponse),
        }

    def default_legacy_task(self) -> str | None:
        return None

    def invoke(self, task: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.predictor is None:
            raise RuntimeError("Kronos model not loaded.")
        if task not in self.task_schemas():
            raise ValueError(f"Kronos does not support task [{task}].")

        import pandas as pd

        if "series" in payload:
            item = (payload.get("series") or [])[0]
            if not item:
                raise ValueError("Kronos request requires at least one series.")
            symbol = item.get("symbol")
            candles = item["candles"]
            horizon = int(payload["horizon"])
            temperature = float(payload.get("temperature") or 1.0)
            top_p = float(payload.get("top_p") or 0.9)
            num_samples = int(payload.get("num_samples") or 1)
        else:
            if task == "generate_paths":
                request = KronosGeneratePathsRequest.model_validate(payload)
                symbol = request.symbol
                candles = [c.model_dump(mode="json", exclude_none=True) for c in request.candles]
                horizon = request.horizon
                temperature = request.temperature
                top_p = request.top_p
                num_samples = request.num_samples
            else:
                request = KronosForecastRequest.model_validate(payload)
                symbol = request.symbol
                candles = [c.model_dump(mode="json", exclude_none=True) for c in request.candles]
                horizon = request.horizon
                temperature = request.temperature
                top_p = request.top_p
                num_samples = 1

        timestamps = [row.get("timestamp") for row in candles]
        x_df = pd.DataFrame(candles)
        x_timestamp = pd.to_datetime(pd.Series(timestamps))
        y_timestamp = self._future_timestamps(x_timestamp, horizon)
        frame = x_df[[c for c in ["open", "high", "low", "close", "volume", "amount"] if c in x_df.columns]]
        frame = frame.dropna(axis=1, how="all")

        if task == "generate_paths":
            paths = []
            for _ in range(num_samples):
                pred_df = self.predictor.predict(
                    df=frame,
                    x_timestamp=x_timestamp,
                    y_timestamp=y_timestamp,
                    pred_len=horizon,
                    T=temperature,
                    top_p=top_p,
                    sample_count=1,
                )
                paths.append(self._dataframe_to_candles(pred_df))
            return {
                "forecasts": [
                    {
                        "symbol": symbol,
                        "paths": paths,
                    }
                ]
            }

        pred_df = self.predictor.predict(
            df=frame,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=horizon,
            T=temperature,
            top_p=top_p,
            sample_count=1,
        )
        return {"forecasts": [{"symbol": symbol, "candles": self._dataframe_to_candles(pred_df)}]}

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
