from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager

import dotenv
import polars as pl
import torch
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Security, UploadFile
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import Settings
from app.mcp import handle_mcp_request
from app.providers import ModelProvider, create_provider
from app.schemas import (
    InvokeRequest,
    InvokeResponse,
    MCPRpcRequest,
    ModelSchemaResponse,
    TimeSeriesInstance,
    UnifiedRequest,
    UnifiedResponse,
)

dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("unitshub")
security = HTTPBearer(auto_error=False)


def create_app(
    settings: Settings | None = None,
    provider: ModelProvider | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if provider is not None:
            app.state.provider = provider
            yield
            return

        model_provider = create_provider(settings)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            "Initializing UniTS-Hub v2 with model=[%s] on device=[%s]",
            settings.model_type,
            device,
        )
        try:
            model_provider.load(settings.model_path(), device)
            logger.info("Model [%s] loaded successfully.", settings.model_type)
        except Exception as exc:
            logger.exception("Failed to load model [%s]: %s", settings.model_type, exc)
        app.state.provider = model_provider
        yield
        app.state.provider = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    app = FastAPI(
        title="UniTS-Hub",
        version=settings.app_version,
        description=(
            "Capability-first serving for TimesFM, Chronos-2, and Kronos. "
            "Provides REST, MCP, and agent skill integration."
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings

    async def get_api_key(
        credentials: HTTPAuthorizationCredentials = Security(security),
    ) -> str:
        token = credentials.credentials if credentials else None
        if token == settings.api_key:
            return token
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")

    def get_provider() -> ModelProvider:
        model_provider = getattr(app.state, "provider", None)
        if model_provider is None:
            raise HTTPException(status_code=503, detail="Model provider is not initialized.")
        if not model_provider.loaded:
            raise HTTPException(
                status_code=503,
                detail=f"Model [{settings.model_type}] is not loaded.",
            )
        return model_provider

    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema

        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        openapi_schema["components"]["securitySchemes"] = {
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "Token",
            }
        }
        openapi_schema["security"] = [{"BearerAuth": []}]
        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi

    @app.get("/health")
    async def health() -> dict:
        current_provider = getattr(app.state, "provider", None)
        return {
            "status": "ready" if current_provider and current_provider.loaded else "not_ready",
            "model": settings.model_type,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "version": settings.app_version,
        }

    @app.get("/models/current")
    async def current_model(
        _: str = Depends(get_api_key),
        current_provider: ModelProvider = Depends(get_provider),
    ) -> dict:
        return current_provider.descriptor().model_dump(mode="json")

    @app.get("/models/current/schema", response_model=ModelSchemaResponse)
    async def current_model_schema(
        _: str = Depends(get_api_key),
        current_provider: ModelProvider = Depends(get_provider),
    ) -> ModelSchemaResponse:
        descriptor = current_provider.descriptor()
        return ModelSchemaResponse(model=descriptor, schemas=current_provider.task_schemas())

    @app.get("/models/current/tasks/{task_name}/schema")
    async def task_schema(
        task_name: str,
        _: str = Depends(get_api_key),
        current_provider: ModelProvider = Depends(get_provider),
    ) -> dict:
        schemas = current_provider.task_schemas()
        if task_name not in schemas:
            raise HTTPException(status_code=404, detail=f"Unknown task [{task_name}].")
        return schemas[task_name]

    @app.post("/models/current/invoke", response_model=InvokeResponse)
    async def invoke_model(
        request: InvokeRequest,
        _: str = Depends(get_api_key),
        current_provider: ModelProvider = Depends(get_provider),
    ) -> InvokeResponse:
        if not current_provider.supports_task(request.task):
            raise HTTPException(status_code=400, detail=f"Task [{request.task}] is not supported.")
        output = current_provider.invoke(request.task, request.input)
        return InvokeResponse(
            model=current_provider.descriptor().id,
            task=request.task,
            output=output,
            metadata={"api": "rest-v2"},
        )

    @app.post("/mcp")
    async def mcp(
        request: MCPRpcRequest,
        _: str = Depends(get_api_key),
        current_provider: ModelProvider = Depends(get_provider),
    ) -> JSONResponse:
        response = handle_mcp_request(current_provider, request.model_dump(mode="json"))
        status_code = 200 if response.error is None else 400
        return JSONResponse(status_code=status_code, content=response.model_dump(mode="json"))

    @app.post("/predict", response_model=UnifiedResponse)
    async def predict(
        request: UnifiedRequest,
        _: str = Depends(get_api_key),
        current_provider: ModelProvider = Depends(get_provider),
    ) -> UnifiedResponse:
        try:
            forecasts = current_provider.legacy_predict(
                history=[instance.history for instance in request.instances],
                horizon=request.task.horizon,
                parameters=request.parameters,
            )
        except NotImplementedError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return UnifiedResponse(
            model=current_provider.descriptor().id,
            forecasts=forecasts,
            metadata={
                "deprecated": True,
                "replacement": "/models/current/invoke",
            },
        )

    @app.post("/predict/csv", response_model=UnifiedResponse)
    async def predict_csv(
        file: UploadFile = File(...),
        target_column: str = Form(...),
        horizon: int = Form(...),
        frequency: str = Form("auto"),
        _: str = Depends(get_api_key),
        current_provider: ModelProvider = Depends(get_provider),
    ) -> UnifiedResponse:
        contents = await file.read()
        try:
            df = pl.read_csv(io.BytesIO(contents))
        except pl.exceptions.NoDataError as exc:
            raise HTTPException(status_code=400, detail="The uploaded CSV file is empty.") from exc
        if target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column [{target_column}] not found.")

        history = df[target_column].drop_nulls().to_list()
        try:
            forecasts = current_provider.legacy_predict(
                history=[history],
                horizon=horizon,
                parameters={"frequency": frequency},
            )
        except NotImplementedError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        _ = TimeSeriesInstance(history=history, metadata={"column": target_column})
        return UnifiedResponse(
            model=current_provider.descriptor().id,
            forecasts=forecasts,
            metadata={
                "deprecated": True,
                "replacement": "/models/current/invoke",
                "source_column": target_column,
            },
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(_: Request, exc: Exception):
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error", "message": str(exc)},
        )

    return app


app = create_app()
