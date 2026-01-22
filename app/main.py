import os
import torch
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import Optional

from app.schemas import UnifiedRequest, UnifiedResponse
from app.core.interface import TimeSeriesModel
from app.core.timesfm import TimesFMEngine
from app.core.chronos import ChronosEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("unitshub")

# Global model instance
MODEL_INSTANCE: Optional[TimeSeriesModel] = None
MODEL_TYPE: str = os.getenv("MODEL_TYPE", "chronos").lower()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL_INSTANCE
    
    # 1. Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🚀 Initializing UniTS-Hub with MODEL_TYPE=[{MODEL_TYPE}] on device=[{device}]")
    
    # 2. Select and Load Model
    try:
        if MODEL_TYPE == "timesfm":
            MODEL_INSTANCE = TimesFMEngine()
            # Path inside Docker container
            model_path = "/app/models/timesfm"
            # In local dev if folder doesn't exist, we might want a fallback or clear error
            MODEL_INSTANCE.load(model_path, device)
        elif MODEL_TYPE == "chronos":
            MODEL_INSTANCE = ChronosEngine()
            model_path = "/app/models/chronos"
            MODEL_INSTANCE.load(model_path, device)
        else:
            raise ValueError(f"Unsupported MODEL_TYPE: {MODEL_TYPE}")
            
        logger.info(f"✅ Model [{MODEL_TYPE}] loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to load model [{MODEL_TYPE}]: {e}")
        # We don't exit here to allow the process to start, but /predict will fail
            
    yield
    
    # Clean up resources
    logger.info("Shutting down UniTS-Hub...")
    MODEL_INSTANCE = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(
    title="UniTS-Hub", 
    description="Unified serving for time-series foundation models",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    status = "ready" if MODEL_INSTANCE else "not_ready"
    return {
        "status": status, 
        "model": MODEL_TYPE,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

@app.post("/predict", response_model=UnifiedResponse)
async def predict(request: UnifiedRequest):
    if not MODEL_INSTANCE:
        raise HTTPException(status_code=503, detail=f"Model [{MODEL_TYPE}] is not initialized or failed to load.")
    
    # Extract inputs
    history_data = [inst.history for inst in request.instances]
    horizon = request.task.horizon
    params = request.parameters or {}
    
    try:
        # Perform inference
        logger.info(f"Processing prediction request: instances={len(history_data)}, horizon={horizon}")
        forecasts = MODEL_INSTANCE.predict(history_data, horizon, **params)
        
        return UnifiedResponse(
            model=MODEL_TYPE,
            forecasts=forecasts
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "message": str(exc)},
    )
