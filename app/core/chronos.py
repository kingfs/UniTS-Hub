import torch
import numpy as np
from chronos import ChronosPipeline
from .interface import TimeSeriesModel
from typing import List, Dict, Any

class ChronosEngine(TimeSeriesModel):
    def __init__(self):
        self.pipeline = None

    def load(self, model_path: str, device: str) -> None:
        """
        Load Chronos model.
        """
        self.pipeline = ChronosPipeline.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        )

    def predict(self, history: List[List[float]], horizon: int, **kwargs) -> List[Dict[str, Any]]:
        if not self.pipeline:
            raise RuntimeError("Chronos model not loaded.")

        # Parameters for Chronos
        num_samples = kwargs.get("num_samples", 20)
        
        # Prepare context
        context = [torch.tensor(h) for h in history]
        
        # Perform prediction
        # forecast shape: (num_series, num_samples, horizon)
        forecast = self.pipeline.predict(
            context, 
            prediction_length=horizon, 
            num_samples=num_samples
        )
        
        results = []
        for sample in forecast.numpy():
            # Calculate median as mean, and common quantiles
            median = np.quantile(sample, 0.5, axis=0)
            p10 = np.quantile(sample, 0.1, axis=0)
            p90 = np.quantile(sample, 0.9, axis=0)
            
            results.append({
                "mean": median.tolist(),
                "quantiles": {
                    "0.1": p10.tolist(),
                    "0.9": p90.tolist()
                }
            })
            
        return results
