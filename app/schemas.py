from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

JsonSchema = Dict[str, Any]


class TaskDefinition(BaseModel):
    name: str = Field(..., description="Stable task identifier.")
    title: str = Field(..., description="Human-readable task name.")
    description: str = Field(..., description="Short description of the task.")
    input_schema: JsonSchema = Field(..., description="JSON Schema for the task input.")
    output_schema: JsonSchema = Field(..., description="JSON Schema for the task output.")


class ModelDescriptor(BaseModel):
    id: str
    name: str
    version: str
    description: str
    input_modes: List[str] = Field(default_factory=list)
    output_modes: List[str] = Field(default_factory=list)
    tasks: List[TaskDefinition] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelSchemaResponse(BaseModel):
    model: ModelDescriptor
    schemas: Dict[str, JsonSchema] = Field(default_factory=dict)


class InvokeRequest(BaseModel):
    task: str = Field(..., description="Task name to invoke.")
    input: Dict[str, Any] = Field(..., description="Task input payload.")


class InvokeResponse(BaseModel):
    model: str
    task: str
    output: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ForecastResult(BaseModel):
    mean: List[float]
    quantiles: Dict[str, List[float]] = Field(default_factory=dict)

    @field_validator("quantiles")
    @classmethod
    def validate_quantile_lengths(cls, value, info):
        mean = info.data.get("mean", [])
        if any(len(series) != len(mean) for series in value.values()):
            raise ValueError("Each quantile forecast must match mean length")
        return value


class TimeSeriesInstance(BaseModel):
    history: List[float] = Field(..., description="Historical values for a single univariate series.")
    metadata: Optional[Dict[str, Any]] = Field(default=None)


class PredictionTask(BaseModel):
    horizon: int = Field(..., gt=0, description="Forecast horizon.")


class UnifiedRequest(BaseModel):
    instances: List[TimeSeriesInstance]
    task: PredictionTask
    parameters: Dict[str, Any] = Field(default_factory=dict)


class UnifiedResponse(BaseModel):
    model: str
    forecasts: List[ForecastResult]
    metadata: Optional[Dict[str, Any]] = None


class TimesFMForecastRequest(BaseModel):
    history: List[float] = Field(..., min_length=1, description="Univariate history.")
    horizon: int = Field(..., gt=0, description="Forecast horizon.")
    frequency: str = Field(default="auto", description="TimesFM frequency bucket or alias.")


class TimesFMForecastResponse(BaseModel):
    mean: List[float]
    quantiles: Dict[str, List[float]] = Field(default_factory=dict)


class QuantileForecastRequest(BaseModel):
    """Canonical single-series quantile request shared by quantile model families."""

    series: List[float] = Field(..., min_length=1, description="Univariate history.")
    horizon: int = Field(..., gt=0, description="Forecast horizon.")
    quantiles: List[float] = Field(
        default_factory=lambda: [0.1, 0.5, 0.9],
        description="Requested quantiles, strictly increasing.",
    )

    @field_validator("quantiles")
    @classmethod
    def validate_quantiles(cls, value: List[float]) -> List[float]:
        if not value or any(not 0 < q < 1 for q in value):
            raise ValueError("quantiles must be non-empty values strictly between 0 and 1")
        if value != sorted(set(value)):
            raise ValueError("quantiles must be strictly increasing")
        return value


class QuantileForecastResponse(BaseModel):
    """Canonical single-series quantile response shared by quantile model families."""

    mean: List[float]
    quantiles: Dict[str, List[float]] = Field(default_factory=dict)


class ChronosForecastRequest(QuantileForecastRequest):
    """Chronos-2 accepts the canonical univariate quantile request unchanged."""


class ChronosForecastResponse(QuantileForecastResponse):
    """Chronos-2 returns the canonical univariate quantile response."""


TIMESTRAL_COVARIATE_FIELDS = (
    "past_covariates",
    "known_covariates",
    "categorical_past_covariates",
    "categorical_known_covariates",
)

TIMESTRAL_COVARIATE_ALIASES = ("future_covariates", "categorical_future_covariates")


def _reject_top_level_covariates(data: Any) -> Any:
    """Reject covariate maps outside ``series[]`` instead of silently dropping them."""
    if isinstance(data, dict):
        found = [name for name in (*TIMESTRAL_COVARIATE_FIELDS, *TIMESTRAL_COVARIATE_ALIASES) if name in data]
        if found:
            raise ValueError(
                "Covariates must be nested under each item in `series`; "
                f"top-level {', '.join(found)} is not supported. "
                "Use forecast_with_covariates and put covariates inside the series item."
            )
    return data


class TimestralUnivariateForecastRequest(QuantileForecastRequest):
    """Timestral single-series request, extending the Chronos-2 contract.

    It adds Timestral's `symmetric_averaging` / `clip_nonnegative` switches.
    `quantiles` must be a subset of the active checkpoint's knots; the runtime
    rejects levels the checkpoint does not expose. A covariate map at the top level
    is rejected instead of being silently dropped by this target-only route.
    """

    symmetric_averaging: bool = Field(
        default=True,
        description="Average the forecast with the reflected forecast of the negated context (upstream default).",
    )
    clip_nonnegative: bool = Field(
        default=True,
        description="Clamp forecasts to non-negative when the observed context is non-negative.",
    )

    @model_validator(mode="before")
    @classmethod
    def _covariates_must_be_nested(cls, data: Any) -> Any:
        return _reject_top_level_covariates(data)


class TimestralUnivariateForecastResponse(QuantileForecastResponse):
    """Timestral single-series response; identical to the Chronos-2 response."""


class Candle(BaseModel):
    timestamp: Optional[str] = None
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None
    amount: Optional[float] = None


class KronosForecastRequest(BaseModel):
    symbol: Optional[str] = Field(default=None)
    candles: List[Candle] = Field(..., min_length=1, description="Historical OHLC/OHLCV candles.")
    horizon: int = Field(..., gt=0, description="Forecast horizon.")
    temperature: float = Field(default=1.0, gt=0.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)


class KronosForecastResponse(BaseModel):
    symbol: Optional[str] = None
    candles: List[Candle]


class KronosGeneratePathsRequest(BaseModel):
    symbol: Optional[str] = Field(default=None)
    candles: List[Candle] = Field(..., min_length=1, description="Historical OHLC/OHLCV candles.")
    horizon: int = Field(..., gt=0, description="Forecast horizon.")
    num_samples: int = Field(..., gt=0, le=64, description="Number of sampled paths.")
    temperature: float = Field(default=1.0, gt=0.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)


class KronosGeneratePathsResponse(BaseModel):
    symbol: Optional[str] = None
    paths: List[List[Candle]]


def _wrap_matrix(value: Any) -> Any:
    """Accept a flat numeric row as a one-row matrix.

    Timestral works on ``(variates, time)`` target and ``(covariates, time)`` covariate
    matrices. Wrapping flat arrays keeps the univariate contract as compact as the other
    models while still exposing the real multivariate shape.
    """
    if value is None or not isinstance(value, (list, tuple)):
        return value
    if not value:
        return []
    if isinstance(value[0], (list, tuple)):
        return [list(row) for row in value]
    return [list(value)]


class TimestralSeriesBase(BaseModel):
    """Target-only series accepted by the `forecast_quantile` task."""

    # Unknown fields are rejected so covariates cannot be silently dropped when
    # the caller should use the `forecast_with_covariates` task instead.
    model_config = ConfigDict(extra="forbid")

    item_id: Optional[str] = Field(default=None, description="Optional caller-supplied series identifier.")
    target: List[List[Optional[float]]] = Field(
        ...,
        description=(
            "Target matrix shaped (variates, time). A flat numeric array is accepted and treated as "
            "one variate. Use null for a missing observation; Timestral handles missing values natively."
        ),
    )
    variate_names: Optional[List[str]] = Field(
        default=None, description="Optional names for the target variates, echoed in the response."
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_target_alias(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if "history" in data and "target" not in data:
            data["target"] = data.pop("history")
        return data

    @field_validator("target", mode="before")
    @classmethod
    def _normalize_target(cls, value: Any) -> Any:
        return _wrap_matrix(value)

    @model_validator(mode="after")
    def _check_target(self) -> "TimestralSeriesBase":
        if not self.target:
            raise ValueError("target must contain at least one variate")
        widths = {len(row) for row in self.target}
        if len(widths) != 1:
            raise ValueError("all variates in target must share the same history length")
        if 0 in widths:
            raise ValueError("target variates must not be empty")
        if any(value is not None and not math.isfinite(value) for row in self.target for value in row):
            raise ValueError("target values must be finite numbers or null")
        if not any(value is not None for row in self.target for value in row):
            raise ValueError("target must contain at least one observed value; null marks a missing value")
        if self.variate_names is not None and len(self.variate_names) != len(self.target):
            raise ValueError("variate_names must match the number of target variates")
        return self

    @property
    def history_length(self) -> int:
        return len(self.target[0])

    @property
    def variate_count(self) -> int:
        return len(self.target)


class TimestralSeries(TimestralSeriesBase):
    """Series accepted by the `forecast_with_covariates` task."""

    past_covariates: Optional[List[List[Optional[float]]]] = Field(
        default=None,
        description="Past-only covariates shaped (covariates, time), covering exactly the context.",
    )
    known_covariates: Optional[List[List[Optional[float]]]] = Field(
        default=None,
        description="Known-future covariates shaped (covariates, time + horizon); the last horizon steps are future.",
    )
    categorical_past_covariates: Optional[List[List[Optional[float]]]] = Field(
        default=None,
        description="Past-only categorical covariates as numeric category codes shaped (covariates, time).",
    )
    categorical_known_covariates: Optional[List[List[Optional[float]]]] = Field(
        default=None,
        description="Known-future categorical covariates as numeric category codes shaped (covariates, time + horizon).",
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_covariate_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        for source, target in (
            ("future_covariates", "known_covariates"),
            ("categorical_future_covariates", "categorical_known_covariates"),
        ):
            if source in data and target not in data:
                data[target] = data.pop(source)
        return data

    @field_validator(*TIMESTRAL_COVARIATE_FIELDS, mode="before")
    @classmethod
    def _normalize_covariates(cls, value: Any) -> Any:
        return _wrap_matrix(value)

    @model_validator(mode="after")
    def _check_covariates(self) -> "TimestralSeries":
        for name in TIMESTRAL_COVARIATE_FIELDS:
            rows = getattr(self, name)
            if not rows:
                continue
            lengths = {len(row) for row in rows}
            if len(lengths) != 1 or 0 in lengths:
                raise ValueError(f"{name} rows must share one non-zero length")
            if any(value is not None and not math.isfinite(value) for row in rows for value in row):
                raise ValueError(f"{name} values must be finite numbers or null")
        return self


class TimestralForecastRequest(BaseModel):
    series: List[TimestralSeriesBase] = Field(..., min_length=1, description="One or more target series.")
    horizon: int = Field(..., gt=0, description="Forecast horizon in time steps.")
    quantiles: List[float] = Field(
        default_factory=lambda: [0.1, 0.5, 0.9],
        description=(
            "Subset of the checkpoint's quantile knots to return. Timestral-209M exposes "
            "0.01, 0.05, 0.1 ... 0.9, 0.95, 0.99; see metadata.available_quantiles."
        ),
    )
    symmetric_averaging: bool = Field(
        default=True,
        description="Average the forecast with the reflected forecast of the negated context (upstream default).",
    )
    clip_nonnegative: bool = Field(
        default=True,
        description="Clamp predictions at zero when the observed context is entirely non-negative.",
    )

    @model_validator(mode="before")
    @classmethod
    def _covariates_must_be_nested(cls, data: Any) -> Any:
        return _reject_top_level_covariates(data)

    @model_validator(mode="before")
    @classmethod
    def _normalize_payload(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if "series" not in data and "history" in data:
            data["series"] = [{"target": data.pop("history")}]
        series = data.get("series")
        if isinstance(series, (list, tuple)):
            if series and not isinstance(series[0], (dict, list, tuple)):
                # Chronos/TimesFM-style flat list: one univariate series.
                data["series"] = [{"target": list(series)}]
            else:
                data["series"] = [
                    {"target": list(item)} if isinstance(item, (list, tuple)) else item for item in series
                ]
        return data

    @field_validator("quantiles")
    @classmethod
    def validate_quantiles(cls, value: List[float]) -> List[float]:
        if not value:
            raise ValueError("quantiles must be non-empty")
        if any(not 0 < quantile < 1 for quantile in value):
            raise ValueError("quantiles must be strictly between 0 and 1")
        if value != sorted(set(value)):
            raise ValueError("quantiles must be strictly increasing")
        return value

    def has_covariates(self) -> bool:
        return False

    def has_known_covariates(self) -> bool:
        return False


class TimestralForecastWithCovariatesRequest(TimestralForecastRequest):
    series: List[TimestralSeries] = Field(..., min_length=1, description="One or more target series with covariates.")

    @model_validator(mode="after")
    def _check_covariates(self) -> "TimestralForecastWithCovariatesRequest":
        for index, item in enumerate(self.series):
            history = item.history_length
            for name in ("past_covariates", "categorical_past_covariates"):
                rows = getattr(item, name)
                if rows and len(rows[0]) != history:
                    raise ValueError(
                        f"series[{index}].{name} must cover exactly the {history} context steps, got {len(rows[0])}"
                    )
            for name in ("known_covariates", "categorical_known_covariates"):
                rows = getattr(item, name)
                expected = history + self.horizon
                if rows and len(rows[0]) != expected:
                    raise ValueError(
                        f"series[{index}].{name} must cover context + horizon ({expected} steps), got {len(rows[0])}"
                    )
        if not self.has_covariates():
            raise ValueError(
                "forecast_with_covariates requires at least one of past_covariates, known_covariates, "
                "categorical_past_covariates or categorical_known_covariates"
            )
        return self

    def has_covariates(self) -> bool:
        return any(bool(getattr(item, name)) for item in self.series for name in TIMESTRAL_COVARIATE_FIELDS)

    def has_known_covariates(self) -> bool:
        return any(
            bool(getattr(item, name))
            for item in self.series
            for name in ("known_covariates", "categorical_known_covariates")
        )


class TimestralForecastResult(BaseModel):
    item_id: Optional[str] = None
    variate_names: Optional[List[str]] = None
    variate_count: int = Field(..., ge=1)
    mean: List[float] | List[List[float]] = Field(
        ...,
        description=(
            "Timestral has no mean head; this is the 0.5 quantile. Flat for a single variate, "
            "nested (variates, horizon) when variate_count > 1."
        ),
    )
    quantiles: Dict[str, List[float] | List[List[float]]] = Field(
        default_factory=dict,
        description="Quantile forecasts keyed by level; flat for one variate, nested otherwise.",
    )


class TimestralForecastResponse(BaseModel):
    forecasts: List[TimestralForecastResult]


def schema_bundle(input_model: type[BaseModel], output_model: type[BaseModel]) -> Dict[str, Any]:
    return {
        "input": input_model.model_json_schema(),
        "output": output_model.model_json_schema(),
    }
