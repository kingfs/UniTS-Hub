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
    schemas: Dict[str, JsonSchema] = Field(
        default_factory=dict,
        description="Map of task name to JSON schema bundle.",
    )


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


class TimeSeriesInstance(BaseModel):
    history: List[float] = Field(
        ...,
        description="Historical values for a single univariate series.",
        examples=[[1.0, 2.0, 3.0, 4.0]],
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional instance metadata such as frequency or series id.",
    )


class PredictionTask(BaseModel):
    horizon: int = Field(..., gt=0, description="Forecast horizon.")


class UnifiedRequest(BaseModel):
    instances: List[TimeSeriesInstance]
    task: PredictionTask
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ForecastResult(BaseModel):
    mean: List[float]
    quantiles: Optional[Dict[str, List[float]]] = None


class UnifiedResponse(BaseModel):
    model: str
    forecasts: List[ForecastResult]
    metadata: Optional[Dict[str, Any]] = None
