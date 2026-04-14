from __future__ import annotations

import json
from typing import Any, Dict

from app.providers.base import ModelProvider
from app.schemas import MCPRpcResponse


SERVER_INFO = {
    "name": "UniTS-Hub MCP",
    "version": "2.0.0",
}


def handle_mcp_request(provider: ModelProvider, request: Dict[str, Any]) -> MCPRpcResponse:
    method = request.get("method")
    params = request.get("params") or {}
    rpc_id = request.get("id")

    try:
        if method == "initialize":
            return MCPRpcResponse(
                id=rpc_id,
                result={
                    "protocolVersion": "2025-03-26",
                    "serverInfo": SERVER_INFO,
                    "capabilities": {
                        "tools": {"listChanged": False},
                        "resources": {"subscribe": False, "listChanged": False},
                    },
                },
            )

        if method == "tools/list":
            descriptor = provider.descriptor()
            tools = [
                {
                    "name": "get_current_model",
                    "description": "Return the currently loaded model descriptor.",
                    "inputSchema": {"type": "object", "properties": {}},
                },
                {
                    "name": "get_model_schema",
                    "description": "Return all task schemas for the current model.",
                    "inputSchema": {"type": "object", "properties": {}},
                },
                {
                    "name": "invoke_model",
                    "description": "Invoke a specific task on the current model.",
                    "inputSchema": {
                        "type": "object",
                        "required": ["task", "input"],
                        "properties": {
                            "task": {
                                "type": "string",
                                "enum": [task.name for task in descriptor.tasks],
                            },
                            "input": {"type": "object"},
                        },
                    },
                },
            ]
            return MCPRpcResponse(id=rpc_id, result={"tools": tools})

        if method == "resources/list":
            return MCPRpcResponse(
                id=rpc_id,
                result={
                    "resources": [
                        {
                            "uri": "unitshub://model/current",
                            "name": "Current model descriptor",
                            "mimeType": "application/json",
                        },
                        {
                            "uri": "unitshub://model/schema",
                            "name": "Current model schemas",
                            "mimeType": "application/json",
                        },
                    ]
                },
            )

        if method == "resources/read":
            uri = params.get("uri")
            if uri == "unitshub://model/current":
                contents = provider.descriptor().model_dump(mode="json")
            elif uri == "unitshub://model/schema":
                contents = provider.task_schemas()
            else:
                raise ValueError(f"Unknown resource [{uri}]")
            return MCPRpcResponse(
                id=rpc_id,
                result={"contents": [{"uri": uri, "mimeType": "application/json", "text": json.dumps(contents)}]},
            )

        if method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments") or {}
            if name == "get_current_model":
                output = provider.descriptor().model_dump(mode="json")
            elif name == "get_model_schema":
                output = provider.task_schemas()
            elif name == "invoke_model":
                output = provider.invoke(arguments["task"], arguments["input"])
            else:
                raise ValueError(f"Unknown tool [{name}]")
            return MCPRpcResponse(
                id=rpc_id,
                result={
                    "content": [{"type": "text", "text": json.dumps(output)}],
                    "structuredContent": output,
                },
            )

        raise ValueError(f"Unsupported MCP method [{method}]")
    except Exception as exc:
        return MCPRpcResponse(
            id=rpc_id,
            error={
                "code": -32000,
                "message": str(exc),
            },
        )
