import os
import torch
import numpy as np
import dotenv
import pytest
from app.core.chronos import ChronosEngine

# Load environment variables
dotenv.load_dotenv()
MODEL_DIR = os.getenv("MODELS_DIR", "/app/models")
MODEL_PATH = os.path.join(MODEL_DIR, "chronos")

@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason=f"Model not found at {MODEL_PATH}")
def test_chronos_inference():
    engine = ChronosEngine()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading Chronos model from {MODEL_PATH} on {device}...")
    engine.load(MODEL_PATH, device=device)
    print("Model loaded successfully.")

    # Sample history: batch size 1, sequence length 32
    history = [np.random.randn(32).tolist()]
    horizon = 12
    
    print(f"Running prediction with horizon={horizon}...")
    results = engine.predict(history, horizon=horizon, num_samples=10)
    print("Prediction successful.")
    
    # Verify first result
    res = results[0]
    assert len(res['mean']) == horizon
    
    if "quantiles" in res and res["quantiles"]:
        print(f"Quantiles found: {list(res['quantiles'].keys())}")
        for q, val in res["quantiles"].items():
            assert len(val) == horizon
    
    print("Chronos Verification PASSED.")
