from __future__ import annotations

from app.config import Settings
from app.providers.base import ModelProvider
from app.providers.chronos import ChronosProvider
from app.providers.kronos import KronosProvider
from app.providers.timesfm import TimesFMProvider


def create_provider(settings: Settings) -> ModelProvider:
    model_type = settings.model_type.lower()
    if model_type == "timesfm":
        return TimesFMProvider()
    if model_type in {"chronos", "chronos2", "chronos-2"}:
        return ChronosProvider()
    if model_type == "kronos":
        return KronosProvider(
            tokenizer_path=settings.kronos_tokenizer_path,
            runtime_path=settings.kronos_runtime_path,
        )
    raise ValueError(f"Unsupported MODEL_TYPE: {settings.model_type}")
