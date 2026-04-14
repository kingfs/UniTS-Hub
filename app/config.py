from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class Settings:
    model_type: str = "chronos"
    models_dir: str = "/app/models"
    api_key: str = "unitshub-secret"
    app_version: str = "2.0.0"
    kronos_tokenizer_path: str | None = None
    kronos_runtime_path: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model_type=os.getenv("MODEL_TYPE", "chronos").lower(),
            models_dir=os.getenv("MODELS_DIR", "/app/models"),
            api_key=os.getenv("API_KEY", "unitshub-secret"),
            app_version=os.getenv("APP_VERSION", "2.0.0"),
            kronos_tokenizer_path=os.getenv("KRONOS_TOKENIZER_PATH"),
            kronos_runtime_path=os.getenv("KRONOS_RUNTIME_PATH"),
        )

    def model_path(self) -> str:
        return os.path.join(self.models_dir, self.model_type)
