import torch
import numpy as np
from chronos import BaseChronosPipeline
from .interface import TimeSeriesModel
from typing import List, Dict, Any

class ChronosEngine(TimeSeriesModel):
    def __init__(self):
        self.pipeline = None

    def load(self, model_path: str, device: str) -> None:
        """
        Load Chronos model. Supports both Chronos v1 and v2 via BaseChronosPipeline.
        """
        self.pipeline = BaseChronosPipeline.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.bfloat16 if device != "cpu" and torch.cuda.is_available() else torch.float32,
        )

    def predict(self, history: List[List[float]], horizon: int, **kwargs) -> List[Dict[str, Any]]:
        if not self.pipeline:
            raise RuntimeError("Chronos model not loaded.")

        # Prepare context
        context = [torch.tensor(h) for h in history]
        
        # Check if it is Chronos 2 (which outputs quantiles) or Chronos v1 (which outputs samples)
        from chronos import ChronosPipeline
        is_v2 = not isinstance(self.pipeline, ChronosPipeline)

        if is_v2:
            # Chronos 2 prediction
            # Returns list of (n_variates, n_quantiles, horizon)
            forecasts = self.pipeline.predict(
                context, 
                prediction_length=horizon
            )
            
            # The quantiles in Chronos 2 are usually [0.01, ..., 0.5, ..., 0.99]
            # We map 0.5 to 'mean', and 0.1/0.9 to 'quantiles'
            all_quantiles = getattr(self.pipeline, "quantiles", [])
            q_map = {q: i for i, q in enumerate(all_quantiles)}
            
            results = []
            for f in forecasts:
                # f shape is (n_variates, n_quantiles, horizon)
                # We assume univariate (n_variates=1)
                f_np = f.squeeze(0).cpu().numpy()
                
                res = {}
                if 0.5 in q_map:
                    res["mean"] = f_np[q_map[0.5]].tolist()
                else:
                    res["mean"] = f_np[len(f_np)//2].tolist() # fallback
                
                res["quantiles"] = {}
                if 0.1 in q_map:
                    res["quantiles"]["0.1"] = f_np[q_map[0.1]].tolist()
                if 0.9 in q_map:
                    res["quantiles"]["0.9"] = f_np[q_map[0.9]].tolist()
                
                results.append(res)
            return results
        else:
            # Chronos v1 prediction (Sample-based)
            num_samples = kwargs.get("num_samples", 20)
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
