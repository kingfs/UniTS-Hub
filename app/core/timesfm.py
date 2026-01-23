from transformers import TimesFmModelForPrediction
import torch
from .interface import TimeSeriesModel
from typing import List, Dict, Any

FREQ_MAP = {
    "auto": 0,
    "high": 0,
    "1h": 0,
    "h": 0,
    "1d": 0,
    "d": 0,
    "medium": 1,
    "1w": 1,
    "w": 1,
    "1m": 1,
    "m": 1,
    "low": 2,
    "1q": 2,
    "q": 2,
    "1y": 2,
    "y": 2,
    "0": 0,
    "1": 1,
    "2": 2
}

class TimesFMEngine(TimeSeriesModel):
    def __init__(self):
        self.model = None
        self.device = None

    def load(self, model_path: str, device: str = "cpu") -> None:
        """
        Load TimesFM 2.5 (PyTorch) model from HuggingFace.
        model_path: HF repo id or local directory.
        """
        self.device = torch.device(device)
        
        # Use float32 on CPU as bfloat16 has limited support for some operations
        dtype = torch.float32 if self.device.type == "cpu" else torch.bfloat16

        # Load config + model
        self.model = TimesFmModelForPrediction.from_pretrained(
            model_path,
            torch_dtype=dtype,
            attn_implementation="sdpa"
        ).to(self.device)

        self.model.eval()

    def predict(
        self,
        history: List[List[float]],
        horizon: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        history: List of time series, each is List[float]
        horizon: forecast horizon
        """
        if self.model is None:
            raise RuntimeError("TimesFM model not loaded.")

        # Convert to tensor: shape [batch, context_length]
        # Use the same dtype as the model to avoid "mat1 and mat2 must have the same dtype" error
        inputs = torch.tensor(history, dtype=self.model.dtype, device=self.device)
        
        # Get frequency index from kwargs (support both 'frequency' and 'freq')
        freq_raw = kwargs.get("frequency") or kwargs.get("freq") or "0"
        if isinstance(freq_raw, str):
            freq_idx = FREQ_MAP.get(freq_raw.lower(), 0)
        else:
            freq_idx = int(freq_raw)
            
        freq = torch.tensor([freq_idx] * len(history), dtype=torch.long, device=self.device)

        # HF TimesFM expects forward call
        with torch.no_grad():
            outputs = self.model(
                past_values=inputs,
                freq=freq,
                return_dict=True
            )
            # mean_predictions shape: [batch, horizon_length]
            mean_predictions = outputs.mean_predictions

        # outputs shape: [batch, horizon]
        # TimesFM 2.0 HF version uses a fixed horizon_length (default 128)
        # We slice it to the requested horizon and cast to float32 for numpy compatibility
        mean_predictions = mean_predictions[:, :horizon].cpu().to(torch.float32).numpy()

        # Standardized output format
        return [
            {
                "mean": row.tolist(),
                "quantiles": {}  # TimesFM 2.5 HF 版暂不提供 quantiles
            }
            for row in mean_predictions
        ]
