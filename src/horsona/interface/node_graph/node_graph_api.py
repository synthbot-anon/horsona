import asyncio
import importlib
import re
from contextlib import asynccontextmanager
from http import HTTPStatus
from time import time
from uuid import uuid4

from fastapi import APIRouter, FastAPI, HTTPException, status
from horsona.autodiff.basic import HorseData, HorseVariable

from .node_graph_models import *

# In-memory storage for sessions and resources
_sessions: dict[str, "Session"] = {}
_session_cleanup_interval = 60
_session_timeout = 300
_session_cleanup_task = None
_allowed_modules = re.compile(r"^(horsona\..*)$")


class Resource(BaseModel):
    id: int
    module_name: str
    class_name: str
    result_obj: Any
    result_dict: Optional[dict[str, Argument]] = None


class Session(BaseModel):
    id: str
    resource_id_to_node: dict[int, Resource] = {}
    resource_obj_to_node: dict[Any, Resource] = {}
    last_active: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    await start_session_cleanup_task()
    yield
    await stop_session_cleanup_task()


router = APIRouter(lifespan=lifespan)


def configure(
    session_timeout: float = 300,
    session_cleanup_interval: float = 60,
    extra_modules: list[str] = [],
):
    """
    Initialize the NodeGraphAPI.

    Args:
        session_timeout (float): The time in seconds after which an inactive session will be removed. Default is 300 seconds.
        session_cleanup_interval (float): The interval in seconds between session cleanup checks. Default is 60 seconds.
        extra_modules (List[str]): A list of additional module names to allow. Default is an empty list.
    """
    global \
        _sessions, \
        _session_timeout, \
        _session_cleanup_interval, \
        _session_cleanup_task, \
        _allowed_modules

    _sessions = {}
    _session_timeout = session_timeout
    _session_cleanup_interval = session_cleanup_interval
    _allowed_modules = re.compile(
        r"^(" + "|".join([r"horsona\..*", *extra_modules]) + ")$"
    )

    if _session_cleanup_task is not None:
        asyncio.create_task(reset_session_cleanup_task())


async def start_session_cleanup_task():
    """
    Start the NodeGraphAPI by initializing the session cleanup task.
    """
    global _session_cleanup_task

    _session_cleanup_task = asyncio.create_task(session_cleanup_task())


async def stop_session_cleanup_task():
    """
    Stop the NodeGraphAPI by cancelling the session cleanup task.
    """
    if _session_cleanup_task is not None:
        _session_cleanup_task.cancel()


async def reset_session_cleanup_task():
    await stop_session_cleanup_task()
    await start_session_cleanup_task()


async def session_cleanup_task():
    """
    Periodically clean up timed-out sessions.
    """
    while True:
        # Wait for the next cleanup
        await asyncio.sleep(_session_cleanup_interval)

        current_time = time()
        sessions_to_remove = [
            session_id
            for session_id, session in _sessions.items()
            if current_time - session.last_active > _session_timeout
        ]

        for session_id in sessions_to_remove:
            del _sessions[session_id]


async def execute(module_name, class_name, function_name, kwargs):
    if not re.match(_allowed_modules, module_name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Module not found"
        )

    module = importlib.import_module(module_name)
    if class_name:
        class_ = getattr(module, class_name)
        if function_name == "__init__":
            function = class_
        else:
            function = getattr(class_, function_name)
    else:
        function = getattr(module, function_name)

    try:
        if asyncio.iscoroutinefunction(function):
            result = await function(**kwargs)
        else:
            result = function(**kwargs)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Failed to execute node",
                "module_name": module_name,
                "class_name": class_name,
                "function_name": function_name,
                "message": str(e),
            },
        )

    return result


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions() -> SessionListResponse:
    """
    Get the list of running sessions.

    Returns:
        SessionListResponse: A response object containing the list of active sessions with their IDs, last active times, and remaining TTLs.
    """
    current_time = time()
    active_sessions = []
    for session_id, session in _sessions.items():
        last_active = session.last_active
        remaining_ttl = max(0, _session_timeout - (current_time - last_active))
        active_sessions.append(
            SessionInfo(
                session_id=session_id,
                last_active=last_active,
                remaining_ttl=remaining_ttl,
            )
        )
    return SessionListResponse(sessions=active_sessions)


@router.get("/docs")
async def get_docs():
    """
    Get the Swagger UI HTML for API documentation.

    Returns:
        HTMLResponse: The Swagger UI HTML.
    """
    return router.get_swagger_ui_html(openapi_url="/openapi.json", title="API Docs")


@router.get("/")
async def root():
    """
    Root endpoint for the API.

    Returns:
        dict: A welcome message.
    """
    return {"message": "Welcome to the Node Graph API"}


@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session() -> CreateSessionResponse:
    """
    Create a new session.

    Returns:
        CreateSessionResponse: An object containing the new session ID and a success message.
    """
    session_id = str(uuid4())
    _sessions[session_id] = Session(id=session_id, last_active=time())
    return CreateSessionResponse(
        session_id=session_id, message="Session created successfully"
    )


@router.post("/sessions/{session_id}/keep_alive", response_model=KeepAliveResponse)
async def keep_alive(session_id: str):
    """
    Keep a session alive by updating its last active time.

    Args:
        session_id (str): The ID of the session to keep alive.

    Returns:
        KeepAliveResponse: A response object with a message confirming the session was kept alive.

    Raises:
        HTTPException: If the session is not found.
    """
    if session_id not in _sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )
    _sessions[session_id].last_active = time()
    return KeepAliveResponse(message="Session kept alive")


@router.get("/sessions/{session_id}/resources", response_model=ListResourcesResponse)
async def list_resources(session_id: str):
    """
    List all resources in a session.

    Args:
        session_id (str): The ID of the session to list resources from.

    Returns:
        ListResourcesResponse: A response object containing a list of all resources in the session.

    Raises:
        HTTPException: If the session is not found.
    """
    if session_id not in _sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )

    resources = []
    for node in _sessions[session_id].resource_id_to_node.values():
        node_id, result_dict = pack_result(session_id, node.result_obj)
        resources.append(
            ResourceResponse(
                id=node_id,
                result=result_dict,
            )
        )
    return ListResourcesResponse(resources=resources)


@router.delete("/sessions/{session_id}", response_model=DeleteSessionResponse)
async def delete_session(session_id: str):
    """
    Delete a session and all its resources.

    Args:
        session_id (str): The ID of the session to delete.

    Returns:
        DeleteSessionResponse: A message confirming the session was deleted.

    Raises:
        HTTPException: If the session is not found.
    """
    if session_id not in _sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )
    del _sessions[session_id]
    return DeleteSessionResponse(
        message=f"Session {session_id} and all its resources deleted successfully"
    )


@router.get(
    "/sessions/{session_id}/resources/{resource_id}", response_model=GetResourceResponse
)
async def get_resource(session_id: str, resource_id: int):
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
    if session_id not in _sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )

    await keep_alive(session_id)

    if resource_id not in _sessions[session_id].resource_id_to_node:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Resource not found in this session",
        )

    node: Resource = _sessions[session_id].resource_id_to_node[resource_id]
    node_id, result_dict = pack_result(session_id, node.result_obj)

    return GetResourceResponse(
        id=node_id,
        result=result_dict,
    )


class InvalidArgumentException(Exception):
    def __init__(self, message: dict):
        self.message = message


def unpack_argument(
    session_id: str, key: list[str], arg: Argument | Any
) -> dict[str, Any]:
    if not isinstance(arg, Argument):
        return arg
    if arg.type == "node":
        return _sessions[session_id].resource_id_to_node[arg.value].result_obj
    elif arg.type in ("str", "float", "int", "bool"):
        return arg.value
    elif arg.type == "list":
        return [
            unpack_argument(session_id, key + [i], item)
            for i, item in enumerate(arg.value)
        ]
    elif arg.type == "dict":
        return {
            k: unpack_argument(session_id, key + [k], v) for k, v in arg.value.items()
        }
    elif arg.type == "tuple":
        return tuple(
            unpack_argument(session_id, key + [i], item)
            for i, item in enumerate(arg.value)
        )
    elif arg.type == "set":
        return {
            unpack_argument(session_id, key + [i], item)
            for i, item in enumerate(arg.value)
        }
    else:
        raise InvalidArgumentException(
            {
                "message": f"Node {arg.value} has invalid argument type: {arg.type}",
                "key": ".".join(key),
                "type": arg.type,
                "value": arg.value,
            }
        )


def create_obj_node(session_id: str, obj: Any) -> Resource:
    if obj in _sessions[session_id].resource_obj_to_node:
        return _sessions[session_id].resource_obj_to_node[obj]

    node_id = len(_sessions[session_id].resource_id_to_node) + 1
    node = Resource(
        id=node_id,
        module_name=obj.__module__,
        class_name=obj.__class__.__name__,
        result_obj=obj,
    )
    _sessions[session_id].resource_id_to_node[node_id] = node
    _sessions[session_id].resource_obj_to_node[obj] = node
    return node


def obj_to_argument(
    session_id: str, key: list[str], obj: Any, recurse=True
) -> Argument:
    if obj is None:
        return Argument(type="none", value=None)
    elif isinstance(obj, (int, float, str, bool)):
        return Argument(type=type(obj).__name__, value=obj)
    elif isinstance(obj, list):
        if recurse:
            return Argument(
                type="list",
                value=[
                    obj_to_argument(session_id, key + [i], item, recurse=False)
                    for i, item in enumerate(obj)
                ],
            )
        else:
            return Argument(type="unsupported", value=None)
    elif isinstance(obj, dict):
        if recurse:
            return Argument(
                type="dict",
                value={
                    k: obj_to_argument(session_id, key + [k], v, recurse=False)
                    for k, v in obj.items()
                },
            )
        else:
            return Argument(type="unsupported", value=None)
    elif isinstance(obj, tuple):
        if recurse:
            return Argument(
                type="tuple",
                value=tuple(
                    obj_to_argument(session_id, key + [i], item, recurse=False)
                    for i, item in enumerate(obj)
                ),
            )
        else:
            return Argument(type="unsupported", value=None)
    elif isinstance(obj, set):
        if recurse:
            return Argument(
                type="set",
                value={
                    obj_to_argument(session_id, key + [i], item, recurse=False)
                    for i, item in enumerate(obj)
                },
            )
        else:
            return Argument(type="unsupported", value=None)
    elif isinstance(obj, HorseData):
        if not recurse:
            node = create_obj_node(session_id, obj)
            return Argument(type="node", value=node.id)

        result_node = create_obj_node(session_id, obj)
        if result_node.result_dict is None:
            result_dict = {}
            for attr_name, attr_value in obj.__dict__.items():
                if isinstance(obj, HorseVariable) and attr_name in (
                    "predecessors",
                    "name",
                    "grad_fn",
                ):
                    continue

                else:
                    result_dict[attr_name] = obj_to_argument(
                        session_id, key + [attr_name], attr_value, recurse=False
                    )

            result_node.result_dict = result_dict

        return Argument(type="node", value=result_node.id)

    else:
        return Argument(type="unsupported", value=None)


def pack_result(session_id, result):
    processed_result = obj_to_argument(session_id, [], result, recurse=True)

    if processed_result.type == "node":
        node = _sessions[session_id].resource_id_to_node[processed_result.value]
        node_id = node.id
        result_dict = node.result_dict
    else:
        node_id = None
        result_dict = processed_result

    return node_id, result_dict


@router.post("/sessions/{session_id}/resources", response_model=PostResourceResponse)
async def post_resource(session_id, request: PostResourceRequest):
    """
    Create a new resource in a session.

    Args:
        session_id (str): The ID of the session to create the resource in.
        request (PostResourceRequest): The request containing session and resource details.

    Returns:
        PostResourceResponse: The details of the newly created resource.

    Raises:
        HTTPException: If the session is not found, the module is not allowed,
                       or there are errors in processing the arguments.
    """
    if session_id not in _sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )

    await keep_alive(session_id)

    processed_kwargs = {}
    errors = []
    for key, arg in request.kwargs.items():
        try:
            processed_kwargs[key] = unpack_argument(session_id, [key], arg)
        except InvalidArgumentException as e:
            errors.append(e.message)

    if errors:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=errors)

    result = await execute(
        request.module_name, request.class_name, request.function_name, processed_kwargs
    )

    node_id, result_dict = pack_result(session_id, result)

    return PostResourceResponse(
        id=node_id,
        result=result_dict,
    )
