import os
import torch
import numpy as np
import dotenv
import pytest
from app.core.timesfm import TimesFMEngine

# Load environment variables
dotenv.load_dotenv()
MODEL_DIR = os.getenv("MODELS_DIR", "/app/models")
MODEL_PATH = os.path.join(MODEL_DIR, "timesfm")

@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason=f"Model not found at {MODEL_PATH}")
def test_timesfm_inference():
    engine = TimesFMEngine()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model {MODEL_PATH} on {device}...")
    engine.load(MODEL_PATH, device=device)
    print("Model loaded successfully.")

    # Sample history: batch size 1, sequence length 32
    history = [np.random.randn(32).tolist()]
    horizon = 12
    
    print(f"Running prediction with horizon={horizon}...")
    results = engine.predict(history, horizon=horizon, freq="0")
    print("Prediction successful.")
    
    assert len(results) == 1
    assert len(results[0]['mean']) == horizon
    print("Verification PASSED.")
