from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class TimeSeriesInstance(BaseModel):
    history: List[float] = Field(..., description="The historical data points for the time series.")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata for this instance.")

class PredictionTask(BaseModel):
    horizon: int = Field(..., gt=0, description="Number of future points to predict.")

class UnifiedRequest(BaseModel):
    instances: List[TimeSeriesInstance]
    task: PredictionTask
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Model-specific parameters (e.g., frequency, num_samples).")

class ForecastResult(BaseModel):
    mean: List[float]
    quantiles: Optional[Dict[str, List[float]]] = None

class UnifiedResponse(BaseModel):
    model: str
    forecasts: List[ForecastResult]
    metadata: Optional[Dict[str, Any]] = None
