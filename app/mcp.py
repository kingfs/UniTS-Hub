from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi.responses import JSONResponse
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.types import ASGIApp, Receive, Scope, Send

from app.providers.base import ModelProvider


class BearerAuthASGI:
    def __init__(self, app: ASGIApp, api_key: str) -> None:
        self.app = app
        self.api_key = api_key

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        auth_header = headers.get(b"authorization", b"").decode("latin-1")
        if auth_header != f"Bearer {self.api_key}":
            response = JSONResponse(status_code=401, content={"detail": "Invalid or missing API key."})
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


def create_mcp_server(get_provider: Callable[[], ModelProvider]) -> FastMCP:
    server = FastMCP(
        name="UniTS-Hub MCP",
        instructions="Discover the active UniTS-Hub model and invoke its supported forecasting tools.",
        stateless_http=True,
        json_response=True,
        streamable_http_path="/",
        # UniTS-Hub mounts the SDK app inside an existing FastAPI service, so host
        # validation should be handled by the outer ingress rather than the inner app.
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )

    @server.tool()
    def get_current_model() -> dict[str, Any]:
        return get_provider().descriptor().model_dump(mode="json")

    @server.tool()
    def get_model_schema() -> dict[str, Any]:
        provider = get_provider()
        return {
            "model": provider.descriptor().model_dump(mode="json"),
            "schemas": provider.task_schemas(),
        }

    @server.tool()
    def get_task_schema(task: str) -> dict[str, Any]:
        provider = get_provider()
        schemas = provider.task_schemas()
        if task not in schemas:
            raise ValueError(f"Unknown task [{task}].")
        return schemas[task]

    @server.tool()
    def invoke_task(task: str, input: dict[str, Any]) -> dict[str, Any]:
        provider = get_provider()
        if not provider.supports_task(task):
            raise ValueError(f"Task [{task}] is not supported.")
        return provider.invoke(task, input)

    return server
