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
                    "name": "get_task_schema",
                    "description": "Return the input/output schema for one task on the current model.",
                    "inputSchema": {
                        "type": "object",
                        "required": ["task"],
                        "properties": {
                            "task": {
                                "type": "string",
                                "enum": [task.name for task in descriptor.tasks],
                            }
                        },
                    },
                },
                {
                    "name": "invoke_task",
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

        if method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments") or {}
            if name == "get_current_model":
                output = provider.descriptor().model_dump(mode="json")
            elif name == "get_task_schema":
                task = arguments["task"]
                output = provider.task_schemas()[task]
            elif name == "invoke_task":
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
