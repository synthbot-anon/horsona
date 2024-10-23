import asyncio
import importlib
import re
from time import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
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
    last_active: float


class Argument(BaseModel):
    type: str
    value: Any


class NodeGraphAPI:
    def __init__(
        self,
        session_timeout: float = 300,
        session_cleanup_interval: float = 60,
        extra_modules: List[str] = [],
    ):
        """
        Initialize the NodeGraphAPI.

        Args:
            session_timeout (float): The time in seconds after which an inactive session will be removed. Default is 300 seconds.
            session_cleanup_interval (float): The interval in seconds between session cleanup checks. Default is 60 seconds.
            extra_modules (List[str]): A list of additional module names to allow. Default is an empty list.
        """
        self.sessions = {}
        self.resources = {}
        self.session_timeout = session_timeout
        self.session_cleanup_interval = session_cleanup_interval
        self.session_cleanup_task = None
        self.allowed_modules = re.compile(
            r"^(" + "|".join([r"horsona\..*", *extra_modules]) + ")$"
        )

        self.app = FastAPI()

        # Define routes
        self.app.get("/")(self.root)
        self.app.post("/sessions")(self.create_session)
        self.app.post("/sessions/{session_id}/keep_alive")(self.keep_alive)
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

    async def start(self):
        """
        Start the NodeGraphAPI by initializing the session cleanup task.
        """
        self.session_cleanup_task = asyncio.create_task(self._cleanup_sessions())

    async def stop(self):
        """
        Stop the NodeGraphAPI by cancelling the session cleanup task.
        """
        if self.session_cleanup_task:
            self.session_cleanup_task.cancel()

    async def _cleanup_sessions(self):
        """
        Periodically clean up timed-out sessions.
        """
        while True:
            current_time = time()
            sessions_to_remove = [
                session_id
                for session_id, session in self.sessions.items()
                if current_time - session.last_active > self.session_timeout
            ]

            for session_id in sessions_to_remove:
                del self.sessions[session_id]

            if sessions_to_remove:
                print(f"Cleaned up {len(sessions_to_remove)} timed-out sessions")

            # Wait for 60 seconds before the next cleanup
            await asyncio.sleep(self.session_cleanup_interval)

    async def get_docs(self):
        """
        Get the Swagger UI HTML for API documentation.

        Returns:
            HTMLResponse: The Swagger UI HTML.
        """
        return self.app.get_swagger_ui_html(
            openapi_url="/openapi.json", title="API Docs"
        )

    async def root(self):
        """
        Root endpoint for the API.

        Returns:
            dict: A welcome message.
        """
        return {"message": "Welcome to the Node Graph API"}

    async def create_session(self):
        """
        Create a new session.

        Returns:
            dict: A dictionary containing the new session ID and a success message.
        """
        session_id = str(uuid4())
        self.sessions[session_id] = Session(id=session_id, last_active=time())
        return {"session_id": session_id, "message": "Session created successfully"}

    async def keep_alive(self, session_id: str):
        """
        Keep a session alive by updating its last active time.

        Args:
            session_id (str): The ID of the session to keep alive.

        Returns:
            dict: A message confirming the session was kept alive.

        Raises:
            HTTPException: If the session is not found.
        """
        if session_id not in self.sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        self.sessions[session_id].last_active = time()
        return {"message": "Session kept alive"}

    async def list_resources(self, session_id: str):
        """
        List all resources in a session.

        Args:
            session_id (str): The ID of the session to list resources from.

        Returns:
            list: A list of all resources in the session.

        Raises:
            HTTPException: If the session is not found.
        """
        if session_id not in self.sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        return list(self.sessions[session_id].resources.values())

    async def delete_session(self, session_id: str):
        """
        Delete a session and all its resources.

        Args:
            session_id (str): The ID of the session to delete.

        Returns:
            dict: A message confirming the session was deleted.

        Raises:
            HTTPException: If the session is not found.
        """
        if session_id not in self.sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        del self.sessions[session_id]
        return {
            "message": f"Session {session_id} and all its resources deleted successfully"
        }

    async def get_resource(self, session_id: str, resource_id: int):
        """
        Get a specific resource from a session.

        Args:
            session_id (str): The ID of the session containing the resource.
            resource_id (int): The ID of the resource to retrieve.

        Returns:
            dict: The details of the requested resource.

        Raises:
            HTTPException: If the session or resource is not found.
        """
        if session_id not in self.sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        await self.keep_alive(session_id)

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
        """
        Create a new resource in a session.

        Args:
            session_id (str): The ID of the session to create the resource in.
            module (str): The name of the module containing the function or class.
            function_name (str): The name of the function to execute.
            class_name (str, optional): The name of the class, if applicable.
            kwargs (Dict[str, Argument]): The arguments to pass to the function.

        Returns:
            dict: The details of the newly created resource.

        Raises:
            HTTPException: If the session is not found, the module is not allowed,
                           or there are errors in processing the arguments.
        """
        if session_id not in self.sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        await self.keep_alive(session_id)

        if not re.match(self.allowed_modules, module):
            raise HTTPException(status_code=404, detail="Module not found")

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
        elif isinstance(result, (int, float, str, bool, list, dict, tuple, set)):
            processed_result = Argument(type=type(result).__name__, value=result)
            node = None
        else:
            node_id = len(self.sessions[session_id].resources) + 1
            processed_result = Argument(type="node", value=node_id)
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
            "id": node.id if node is not None else None,
            "result": processed_result,
        }

    async def delete_resource(self, session_id: str, resource_id: str):
        """
        Delete a specific resource from a session.

        Args:
            session_id (str): The ID of the session containing the resource.
            resource_id (str): The ID of the resource to delete.

        Returns:
            dict: A message confirming the resource was deleted.

        Raises:
            HTTPException: If the session or resource is not found.
        """
        if session_id not in self.sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        await self.keep_alive(session_id)

        if resource_id not in self.sessions[session_id].resources:
            raise HTTPException(
                status_code=404, detail="Resource not found in this session"
            )
        del self.sessions[session_id].resources[resource_id]
        return {
            "message": f"Resource {resource_id} deleted successfully from session {session_id}"
        }
