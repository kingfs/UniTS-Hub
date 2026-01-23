from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class TimeSeriesInstance(BaseModel):
    history: List[float] = Field(
        ..., 
        description="The historical data points for the time series. Must be a list of numerical values.",
        example=[0.1, 0.2, 0.3, 0.4, 0.5]
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, 
        description="Optional metadata for this instance, such as sequence ID or tags."
    )

class PredictionTask(BaseModel):
    horizon: int = Field(
        ..., 
        gt=0, 
        description="Number of future points to predict. Must be a positive integer.",
        example=12
    )

class UnifiedRequest(BaseModel):
    instances: List[TimeSeriesInstance] = Field(
        ..., 
        description="List of time series instances to perform prediction on. Batch processing is supported."
    )
    task: PredictionTask = Field(
        ..., 
        description="Configuration for the prediction task, primarily the prediction horizon."
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Model-specific parameters. Common keys include 'freq' (frequency, e.g., 'H', 'D') and 'num_samples' (for probabilistic forecasting).",
        example={"freq": "H", "num_samples": 20}
    )

class ForecastResult(BaseModel):
    mean: List[float] = Field(..., description="The predicted point forecasts (mean values).")
    quantiles: Optional[Dict[str, List[float]]] = Field(
        None, 
        description="Probabilistic forecasts at different quantiles (e.g., '0.1', '0.9')."
    )

class UnifiedResponse(BaseModel):
    model: str = Field(..., description="The name of the model used for prediction (e.g., 'chronos', 'timesfm').")
    forecasts: List[ForecastResult] = Field(..., description="List of forecast results corresponding to the input instances.")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata related to the prediction execution.")
