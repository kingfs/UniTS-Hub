from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


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


class MCPRpcRequest(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: Optional[Any] = None
    method: str
    params: Dict[str, Any] = Field(default_factory=dict)


class MCPRpcResponse(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: Optional[Any] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


class ForecastResult(BaseModel):
    mean: List[float]
    quantiles: Dict[str, List[float]] = Field(default_factory=dict)


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


class ChronosForecastRequest(BaseModel):
    series: List[float] = Field(..., min_length=1, description="Univariate history for Chronos-2.")
    horizon: int = Field(..., gt=0, description="Forecast horizon.")
    quantiles: List[float] = Field(
        default_factory=lambda: [0.1, 0.5, 0.9],
        description="Requested quantiles.",
    )


class ChronosForecastResponse(BaseModel):
    mean: List[float]
    quantiles: Dict[str, List[float]] = Field(default_factory=dict)


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


def schema_bundle(input_model: type[BaseModel], output_model: type[BaseModel]) -> Dict[str, Any]:
    return {
        "input": input_model.model_json_schema(),
        "output": output_model.model_json_schema(),
    }
