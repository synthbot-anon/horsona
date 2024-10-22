import asyncio
import importlib
import re
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from horsona.autodiff.basic import HorseData
from pydantic import BaseModel

# In-memory storage for sessions and resources
sessions = {}
resources = {}


class Node(BaseModel):
    id: int
    module_name: str
    class_name: Optional[str] = None
    function_name: Optional[str] = None
    kwargs: Optional[Dict[str, Any]] = None
    result: Any
    processed_result: Dict[str, Any] = None
    result_node: Optional["Node"] = None

    def __init__(self, **data):
        super().__init__(**data)


async def execute(module_name, class_name, function_name, kwargs):
    module = importlib.import_module(module_name)
    if class_name:
        class_ = getattr(module, class_name)
        if function_name == "__init__":
            function = class_
        else:
            function = getattr(class_, function_name)
    else:
        function = getattr(module, function_name)

    if asyncio.iscoroutinefunction(function):
        result = await function(**kwargs)
    else:
        result = function(**kwargs)

    return result


class Session(BaseModel):
    id: str
    resources: dict[int, Node] = {}
    resource_to_node: dict[Any, Node] = {}


class Argument(BaseModel):
    type: str
    value: Any


class NodeGraphAPI:
    def __init__(self, extra_modules: List[str] = []):
        self.sessions = {}
        self.resources = {}
        self.allowed_modules = re.compile(
            r"^(" + "|".join([r"horsona\..*", *extra_modules]) + ")$"
        )

        self.app = FastAPI()
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Allows all origins
            allow_credentials=True,
            allow_methods=["*"],  # Allows all methods
            allow_headers=["*"],  # Allows all headers
        )

        # Define routes
        self.app.get("/")(self.root)
        self.app.post("/sessions")(self.create_session)
        self.app.get("/sessions/{session_id}/resources")(self.list_resources)
        self.app.delete("/sessions/{session_id}")(self.delete_session)
        self.app.get("/sessions/{session_id}/resources/{resource_id}")(
            self.get_resource
        )
        self.app.post("/sessions/{session_id}/resources")(self.post_resource)
        self.app.delete("/sessions/{session_id}/resources/{resource_id}")(
            self.delete_resource
        )
        self.app.get("/docs", include_in_schema=False)(self.get_docs)

    async def get_docs(self):
        return self.app.get_swagger_ui_html(
            openapi_url="/openapi.json", title="API Docs"
        )

    async def root(self):
        return {"message": "Welcome to the Node Graph API"}

    async def create_session(self):
        session_id = str(uuid4())
        self.sessions[session_id] = Session(id=session_id)
        return {"session_id": session_id, "message": "Session created successfully"}

    async def list_resources(self, session_id: str):
        if session_id not in self.sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        return list(self.sessions[session_id].resources.values())

    async def delete_session(self, session_id: str):
        if session_id not in self.sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        del self.sessions[session_id]
        return {
            "message": f"Session {session_id} and all its resources deleted successfully"
        }

    async def get_resource(self, session_id: str, resource_id: int):
        if session_id not in self.sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        if resource_id not in self.sessions[session_id].resources:
            raise HTTPException(
                status_code=404, detail="Resource not found in this session"
            )
        node: Node = self.sessions[session_id].resources[resource_id]

        return {
            "id": node.id,
            "module_name": node.module_name,
            "class_name": node.class_name,
            "function_name": node.function_name,
            "result": node.processed_result,
        }

    async def post_resource(
        self,
        session_id: str,
        module: str,
        function_name: str,
        class_name: str = None,
        kwargs: Dict[str, Argument] = {},
    ):
        if session_id not in self.sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        processed_kwargs = {}
        errors = []
        for key, arg in kwargs.items():
            try:
                if arg.type == "node":
                    if arg.value not in self.sessions[session_id].resources:
                        errors.append(
                            {
                                "message": f"Node {arg.value} not found in this session",
                                "key": key,
                                "type": arg.type,
                                "value": arg.value,
                            }
                        )
                    else:
                        processed_kwargs[key] = (
                            self.sessions[session_id].resources[arg.value].result
                        )
                elif arg.type == "int":
                    processed_kwargs[key] = int(arg.value)
                elif arg.type == "float":
                    processed_kwargs[key] = float(arg.value)
                elif arg.type == "str":
                    processed_kwargs[key] = str(arg.value)
                elif arg.type == "bool":
                    processed_kwargs[key] = bool(arg.value)
                elif arg.type == "list":
                    processed_kwargs[key] = list(arg.value)
                elif arg.type == "dict":
                    processed_kwargs[key] = dict(arg.value)
                elif arg.type == "tuple":
                    processed_kwargs[key] = tuple(arg.value)
                elif arg.type == "set":
                    processed_kwargs[key] = set(arg.value)
                else:
                    errors.append(
                        {
                            "error": f"Invalid argument type",
                            "key": key,
                            "type": arg.type,
                            "value": arg.value,
                        }
                    )
            except Exception as e:
                errors.append(
                    {
                        "error": f"Invalid argument value",
                        "key": key,
                        "type": arg.type,
                        "value": arg.value,
                    }
                )

        if errors:
            raise HTTPException(status_code=400, detail=errors)

        try:
            result = await execute(module, class_name, function_name, processed_kwargs)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Failed to execute node",
                    "module_name": module,
                    "class_name": class_name,
                    "function_name": function_name,
                    "message": str(e),
                },
            )

        if result in self.sessions[session_id].resource_to_node:
            node = self.sessions[session_id].resource_to_node[result]
            processed_result = node.processed_result
        elif isinstance(result, HorseData):
            processed_result = {}
            for attr_name, attr_value in result.__dict__.items():
                if attr_name in ("predecessors", "name"):
                    continue
                elif isinstance(attr_value, (int, float, str, bool)):
                    attr_type = type(attr_value).__name__
                    processed_result[attr_name] = Argument(
                        type=attr_type, value=attr_value
                    )
                    continue
                elif isinstance(attr_value, HorseData):
                    if attr_value in self.sessions[session_id].resource_to_node:
                        processed_result[attr_name] = Argument(
                            type="node",
                            value=self.sessions[session_id]
                            .resource_to_node[attr_value]
                            .id,
                        )
                        continue

                    attr_module_name = type(attr_value).__module__
                    attr_class_name = type(attr_value).__name__

                    attr_node_id = len(self.sessions[session_id].resources) + 1
                    attr_node = Node(
                        id=attr_node_id,
                        module_name=attr_module_name,
                        class_name=attr_class_name,
                        function_name=None,
                        kwargs=None,
                        result=attr_value,
                    )
                    self.sessions[session_id].resources[attr_node_id] = attr_node
                    self.sessions[session_id].resource_to_node[attr_value] = attr_node
                    processed_result[attr_name] = Argument(
                        type="node", value=attr_node_id
                    )

            node_id = len(self.sessions[session_id].resources) + 1
            node = Node(
                id=node_id,
                module_name=module,
                class_name=class_name,
                function_name=function_name,
                kwargs=kwargs,
                result=result,
                processed_result=processed_result,
            )

            self.sessions[session_id].resources[node_id] = node
            self.sessions[session_id].resource_to_node[result] = node

        return {
            "id": node.id,
            "result": processed_result,
        }

    async def delete_resource(self, session_id: str, resource_id: str):
        if session_id not in self.sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        if resource_id not in self.sessions[session_id].resources:
            raise HTTPException(
                status_code=404, detail="Resource not found in this session"
            )
        del self.sessions[session_id].resources[resource_id]
        return {
            "message": f"Resource {resource_id} deleted successfully from session {session_id}"
        }
