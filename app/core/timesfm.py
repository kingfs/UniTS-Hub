from transformers import TimesFmModelForPrediction
import torch
from .interface import TimeSeriesModel
from typing import List, Dict, Any

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

        # Load config + model
        self.model = TimesFmModelForPrediction.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
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
        inputs = torch.tensor(history, dtype=torch.float32, device=self.device)

        # HF TimesFM expects shape [batch, context_length]
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                prediction_length=horizon,
                num_samples=1,  # deterministic forecast
            )

        # outputs shape: [batch, horizon]
        outputs = outputs.cpu().numpy()

        # Standardized output format
        return [
            {
                "mean": row.tolist(),
                "quantiles": {}  # TimesFM 2.5 HF 版暂不提供 quantiles
            }
            for row in outputs
        ]
