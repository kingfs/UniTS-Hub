from __future__ import annotations

import io
import json
import logging
from contextlib import AsyncExitStack, asynccontextmanager
from typing import TypeVar

import dotenv
import polars as pl
import torch
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Security, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.mcp import BearerAuthASGI, create_mcp_server
from app.providers import ModelProvider, create_provider
from app.schemas import (
    ChronosForecastRequest,
    ChronosForecastResponse,
    InvokeRequest,
    InvokeResponse,
    KronosForecastRequest,
    KronosForecastResponse,
    KronosGeneratePathsRequest,
    KronosGeneratePathsResponse,
    ModelSchemaResponse,
    TimeSeriesInstance,
    TimesFMForecastRequest,
    TimesFMForecastResponse,
    UnifiedRequest,
    UnifiedResponse,
)

dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("unitshub")
security = HTTPBearer(auto_error=False)
BodyModel = TypeVar("BodyModel", bound=BaseModel)


def create_app(
    settings: Settings | None = None,
    provider: ModelProvider | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    mcp_server = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with AsyncExitStack() as stack:
            if mcp_server is not None:
                await stack.enter_async_context(mcp_server.session_manager.run())

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

    async def parse_json_body(request: Request, model: type[BodyModel]) -> BodyModel:
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Request body must be valid JSON.") from exc

        if not isinstance(payload, dict):
            raise HTTPException(status_code=422, detail="Request body must be a JSON object.")

        try:
            return model.model_validate(payload)
        except ValidationError as exc:
            raise RequestValidationError(exc.errors(), body=payload) from exc

    def require_model(current_provider: ModelProvider, model_id: str) -> None:
        active_model = current_provider.descriptor().id
        if active_model != model_id:
            raise HTTPException(
                status_code=404,
                detail=f"Route is only available when model [{model_id}] is active. Current model is [{active_model}].",
            )

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
    mcp_server = create_mcp_server(get_provider)
    app.mount("/mcp", BearerAuthASGI(mcp_server.streamable_http_app(), settings.api_key))

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
        raw_request: Request,
        _: str = Depends(get_api_key),
        current_provider: ModelProvider = Depends(get_provider),
    ) -> InvokeResponse:
        request = await parse_json_body(raw_request, InvokeRequest)
        if not current_provider.supports_task(request.task):
            raise HTTPException(status_code=400, detail=f"Task [{request.task}] is not supported.")
        output = current_provider.invoke(request.task, request.input)
        return InvokeResponse(
            model=current_provider.descriptor().id,
            task=request.task,
            output=output,
            metadata={"api": "rest-v2"},
        )

    @app.post("/timesfm/forecast", response_model=TimesFMForecastResponse)
    async def timesfm_forecast(
        raw_request: Request,
        _: str = Depends(get_api_key),
        current_provider: ModelProvider = Depends(get_provider),
    ) -> TimesFMForecastResponse:
        request = await parse_json_body(raw_request, TimesFMForecastRequest)
        require_model(current_provider, "timesfm")
        output = current_provider.invoke("forecast_point", request.model_dump(mode="json"))
        forecast = output["forecasts"][0]
        return TimesFMForecastResponse(mean=forecast["mean"])

    @app.post("/chronos/forecast", response_model=ChronosForecastResponse)
    async def chronos_forecast(
        raw_request: Request,
        _: str = Depends(get_api_key),
        current_provider: ModelProvider = Depends(get_provider),
    ) -> ChronosForecastResponse:
        request = await parse_json_body(raw_request, ChronosForecastRequest)
        require_model(current_provider, "chronos")
        output = current_provider.invoke("forecast_quantile", request.model_dump(mode="json"))
        forecast = output["forecasts"][0]
        return ChronosForecastResponse(mean=forecast["mean"], quantiles=forecast.get("quantiles") or {})

    @app.post("/kronos/forecast-ohlcv", response_model=KronosForecastResponse)
    async def kronos_forecast_ohlcv(
        raw_request: Request,
        _: str = Depends(get_api_key),
        current_provider: ModelProvider = Depends(get_provider),
    ) -> KronosForecastResponse:
        request = await parse_json_body(raw_request, KronosForecastRequest)
        require_model(current_provider, "kronos")
        output = current_provider.invoke("forecast_ohlcv", request.model_dump(mode="json"))
        forecast = output["forecasts"][0]
        return KronosForecastResponse.model_validate(forecast)

    @app.post("/kronos/generate-paths", response_model=KronosGeneratePathsResponse)
    async def kronos_generate_paths(
        raw_request: Request,
        _: str = Depends(get_api_key),
        current_provider: ModelProvider = Depends(get_provider),
    ) -> KronosGeneratePathsResponse:
        request = await parse_json_body(raw_request, KronosGeneratePathsRequest)
        require_model(current_provider, "kronos")
        output = current_provider.invoke("generate_paths", request.model_dump(mode="json"))
        forecast = output["forecasts"][0]
        return KronosGeneratePathsResponse.model_validate(forecast)

    @app.post("/predict", response_model=UnifiedResponse)
    async def predict(
        raw_request: Request,
        _: str = Depends(get_api_key),
        current_provider: ModelProvider = Depends(get_provider),
    ) -> UnifiedResponse:
        request = await parse_json_body(raw_request, UnifiedRequest)
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
