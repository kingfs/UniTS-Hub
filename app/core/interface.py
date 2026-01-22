from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
import numpy as np

class TimeSeriesModel(ABC):
    """
    Abstract base class for all time-series model adapters.
    """
    
    @abstractmethod
    def load(self, model_path: str, device: str) -> None:
        """
        Load model weights into memory/GPU.
        """
        pass

    @abstractmethod
    def predict(self, history: List[List[float]], horizon: int, **kwargs) -> List[Dict[str, Any]]:
        """
        Perform inference on provided history sequences.
        Returns a list of forecasts (one per input sequence).
        """
        pass
