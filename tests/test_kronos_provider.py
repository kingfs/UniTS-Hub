from __future__ import annotations

import pandas as pd

from app.providers.kronos import KronosProvider


class RecordingPredictor:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        pred_len = int(kwargs["pred_len"])
        y_timestamp = kwargs["y_timestamp"]
        df = kwargs["df"]
        base = {
            "open": [1.0] * pred_len,
            "high": [1.0] * pred_len,
            "low": [1.0] * pred_len,
            "close": [1.0] * pred_len,
        }
        if "volume" in df.columns:
            base["volume"] = [1.0] * pred_len
        if "amount" in df.columns:
            base["amount"] = [1.0] * pred_len
        return pd.DataFrame(base, index=y_timestamp)


def test_kronos_omits_optional_amount_column_when_missing():
    provider = KronosProvider()
    predictor = RecordingPredictor()
    provider.predictor = predictor

    provider.invoke(
        "forecast_ohlcv",
        {
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
            ],
            "horizon": 2,
        },
    )

    frame = predictor.calls[0]["df"]
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
