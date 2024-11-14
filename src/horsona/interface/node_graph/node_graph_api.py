import abc
import asyncio
import importlib
import inspect
import pkgutil
import re
import types
import typing
from contextlib import asynccontextmanager
from inspect import signature
from time import time
from types import NoneType, UnionType
from uuid import uuid4

from fastapi import APIRouter, Body, FastAPI, HTTPException, Request, status

from horsona.autodiff.basic import HorseData, HorseVariable

from .node_graph_models import *

# In-memory storage for sessions and resources
_sessions: dict[str, "Session"] = {}
_session_cleanup_interval = 60
_session_timeout = 300
_session_cleanup_task = None
_allowed_modules_regex = re.compile(r"^(horsona\..*)$")
_allowed_modules: list[str] = []
_allowed_module_names: set[str] = set()


class Resource(BaseModel):
    id: int
    module_name: str
    class_name: str
    result_obj: Any


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


router = APIRouter(prefix="/api", lifespan=lifespan)


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
        _allowed_modules, \
        _allowed_module_names, \
        _allowed_modules_regex

    _sessions = {}
    _session_timeout = session_timeout
    _session_cleanup_interval = session_cleanup_interval
    _allowed_modules_regex = re.compile(
        (r"^(" + "|".join([rf"{x}$|{x}\..*$" for x in _allowed_modules]) + ")")
    )

    _allowed_module_names = set()
    _allowed_modules = set()
    for parent_module_name in ["horsona", *extra_modules]:
        module = importlib.import_module(parent_module_name)
        _allowed_modules.add(module)
        _allowed_module_names.add(parent_module_name)
        if hasattr(module, "__path__"):
            for loader, module_name, is_pkg in pkgutil.walk_packages(
                module.__path__, parent_module_name + "."
            ):
                _allowed_modules.add(importlib.import_module(module_name))
                _allowed_module_names.add(module_name)
    if _session_cleanup_task is not None:
        asyncio.create_task(reset_session_cleanup_task())


def _get_param_annotation(annotation) -> type:
    """
    Convert Python type annotations into corresponding Argument types for the node graph.

    Args:
        annotation: The Python type annotation to convert

    Returns:
        type: The corresponding Argument type for the node graph

    This function handles:
    - Basic types (int, float, bool, str)
    - Container types (list, dict, tuple, set)
    - Union and Optional types
    - Type variables and forward references
    - Class types (converted to NodeArgument)
    """
    # Handle Union types (e.g. Union[str, int])
    if isinstance(annotation, types.UnionType):
        return Union[tuple(_get_param_annotation(arg) for arg in annotation.__args__)]

    # Get the origin type for generic types
    if hasattr(annotation, "__origin__"):
        origin = annotation.__origin__
    elif hasattr(annotation, "__bound__"):
        return _get_param_annotation(annotation.__bound__)
    else:
        origin = annotation

    # Map Python types to Argument types
    if origin == int:
        return IntArgument
    elif origin == float:
        return FloatArgument
    elif origin == bool:
        return BoolArgument
    elif origin == str:
        return StrArgument
    elif origin == list:
        return ListArgument
    elif origin == dict:
        return DictArgument
    elif origin == tuple:
        return TupleArgument
    elif origin == set:
        return SetArgument
    elif origin == None:
        return NoneArgument
    elif origin == Union:
        return Union[tuple(_get_param_annotation(arg) for arg in annotation.__args__)]
    elif origin == Optional:
        return Optional[_get_param_annotation(annotation.__args__[0])]
    elif origin == typing.Type:
        return _get_param_annotation(annotation.__args__[0])
    elif origin == typing.TypeVar:
        return _get_param_annotation(annotation.__bound__)
    elif origin == type:
        return _get_param_annotation(annotation.__args__[0])
    elif inspect.isclass(origin):
        return NodeArgument
    elif isinstance(origin, str):
        return NodeArgument
    elif origin == typing.Any:
        return Argument
    else:
        return None


class DependencyOverride:
    use_cache: bool = False


def _create_route(app: FastAPI, path: str, method_obj: Any) -> dict[str, Any]:
    """
    Create a FastAPI route for a given method with appropriate argument types.

    Args:
        app: The FastAPI application to add the route to
        path: The URL path for the route
        method_obj: The method object to create a route for

    Returns:
        dict: Route configuration if successful, None if route creation failed

    This function:
    1. Extracts parameter and return type annotations from the method
    2. Converts Python types to Argument types
    3. Creates a new method with the converted types
    4. Adds the route to the FastAPI app
    """
    # Get original method specifications
    orig_spec = inspect.getfullargspec(method_obj)
    orig_params = orig_spec.args
    orig_annotations = method_obj.__annotations__

    # Convert annotations to Argument types
    new_annotations = {}
    for param in orig_params:
        # Handle self parameter for class methods
        if param == "self" and orig_annotations.get(param) == None:
            new_annotation = NodeArgument
            new_annotations["self"] = NodeArgument
            continue

        # Handle missing annotations
        if param not in orig_annotations:
            if orig_spec.varkw == param:
                new_annotation = DictArgument
                new_annotations[param] = DictArgument
                continue
            print(f"Skipping {path} due to missing annotation for parameter {param}")
            return None

        # Convert annotation to Argument type
        new_annotation = _get_param_annotation(orig_annotations[param])
        if new_annotation is None:
            print(
                f"Skipping {path} due to unsupported annotation for parameter {param}"
            )
            return None

        new_annotations[param] = new_annotation

    # Handle return type annotation
    if "return" not in orig_annotations:
        if method_obj.__name__ != "__init__":
            print(f"Skipping {path} due to missing return annotation")
            return None
        else:
            new_annotations["return"] = NodeArgument
    else:
        new_annotations["return"] = _get_param_annotation(orig_annotations["return"])

    response_model = new_annotations.pop("return")

    # Create the new method
    args = "\n    ".join(
        [
            f"{f}: {v.__repr__() if type(v).__module__ == 'typing' else v.__name__}"
            for f, v in new_annotations.items()
        ]
    )
    args_name = "_".join(path.split("/")[-1].split("."))
    if args:
        ns = {x.__name__: x for x in _allowed_modules}
        ns.update(globals())
        exec(
            (
                f"class Args_{args_name}(BaseModel):\n"
                f"    {args}\n"
                f"\n"
                f"def new_method(session_id: str, args: Args_{args_name}):\n"
                f"    pass\n"
            ),
            ns,
            ns,
        )

        new_method = ns["new_method"]
    else:

        def new_method(session_id: str):
            pass

    new_method.__doc__ = method_obj.__doc__

    # Add the new method to the FastAPI app
    app.post(path, response_model=response_model)(new_method)


@router.get("/docs")
async def docs():
    from fastapi.openapi.docs import get_swagger_ui_html

    # return swagger ui
    return get_swagger_ui_html(
        openapi_url="/api/openapi.json", title="Horsona Node Graph API"
    )


@router.get("/openapi.json")
async def get_openapi():
    """
    Generate OpenAPI specification for all allowed modules and their functions/methods.

    Returns:
        dict: OpenAPI specification document

    This endpoint:
    1. Creates a temporary FastAPI app
    2. Scans all allowed modules for functions and classes
    3. Creates routes for each function/method with appropriate type conversions
    4. Generates OpenAPI spec from the routes
    """
    import inspect

    from fastapi.openapi.utils import get_openapi

    # Create temporary FastAPI app to generate OpenAPI spec
    temp_app = FastAPI()

    # Scan all allowed modules
    for module in _allowed_modules:
        # Get all functions and classes in module
        for name, obj in inspect.getmembers(module):
            if not hasattr(obj, "__module__"):
                continue
            if obj.__module__ != module.__name__:
                continue

            if name.startswith("_"):
                continue

            # Handle standalone functions
            if inspect.isfunction(obj) and not type(obj) == type:
                path = f"{router.prefix}/sessions/{{session_id}}/resources/{module.__name__}/{name}"
                _create_route(temp_app, path, obj)

            # Handle classes and their methods
            elif inspect.isclass(obj):
                for method_name, method in inspect.getmembers(
                    obj, predicate=inspect.isfunction
                ):
                    path = f"{router.prefix}/sessions/{{session_id}}/resources/{module.__name__}/{name}.{method_name}"
                    if method.__module__ not in _allowed_module_names:
                        continue

                    # Skip abstract methods
                    if (
                        hasattr(obj, "__abstractmethods__")
                        and method_name in obj.__abstractmethods__
                    ):
                        continue

                    # Handle special methods
                    if method_name.startswith("_"):
                        if method_name != "__init__":
                            continue
                        if (
                            hasattr(obj, "__abstractmethods__")
                            and obj.__abstractmethods__
                        ):
                            continue
                    if not hasattr(method, "__call__"):
                        continue

                    _create_route(temp_app, path, method)

    return get_openapi(
        title="Horsona Node Graph API",
        version="0.1.0",
        routes=temp_app.routes,
    )


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
    if module_name not in _allowed_module_names:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Module not found"
        )

    try:
        module = importlib.import_module(module_name)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Module not found"
        )

    try:
        if class_name:
            class_ = getattr(module, class_name)
            if function_name == "__init__":
                function = class_
            else:
                if isinstance(getattr(class_, function_name), classmethod):
                    function = getattr(class_, function_name)
                else:
                    function = getattr(kwargs["self"], function_name)
                    del kwargs["self"]
        else:
            function = getattr(module, function_name)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Function not found"
        )

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
        data, result = pack_result(session_id, [], node.result_obj, recurse=True)
        resources.append(
            ResourceResponse(
                result=result,
                data=data,
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
    data, result = pack_result(session_id, [], node.result_obj, recurse=True)

    return GetResourceResponse(
        result=result,
        data=data,
    )


class InvalidArgumentException(Exception):
    def __init__(self, message: dict):
        self.message = message


def unpack_argument(
    session_id: str, key: list[str], arg: Argument | Any
) -> dict[str, Any]:
    if arg.type == ArgumentType.NODE:
        return _sessions[session_id].resource_id_to_node[arg.value].result_obj
    elif arg.type in (
        ArgumentType.STR,
        ArgumentType.FLOAT,
        ArgumentType.INT,
        ArgumentType.BOOL,
    ):
        return arg.value
    elif arg.type == ArgumentType.LIST:
        return [
            unpack_argument(session_id, key + [i], item)
            for i, item in enumerate(arg.value)
        ]
    elif arg.type == ArgumentType.DICT:
        return {
            k: unpack_argument(session_id, key + [k], v) for k, v in arg.value.items()
        }
    elif arg.type == ArgumentType.TUPLE:
        return tuple(
            unpack_argument(session_id, key + [i], item)
            for i, item in enumerate(arg.value)
        )
    elif arg.type == ArgumentType.SET:
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


def pack_result(
    session_id: str, key: list[str], obj: Any, recurse=True
) -> tuple[Optional[int], Argument | dict[str, Argument]]:
    if obj is None:
        return None, NoneArgument(type="none", value=None)
    elif isinstance(obj, (int, float, str, bool)):
        return None, create_argument(type=type(obj).__name__, value=obj)
    elif isinstance(obj, list):
        if recurse:
            return None, ListArgument(
                type="list",
                value=[
                    pack_result(session_id, key + [i], item, recurse=False)[1]
                    for i, item in enumerate(obj)
                ],
            )
        else:
            return None, UnsupportedArgument(type="unsupported", value=None)
    elif isinstance(obj, dict):
        if recurse:
            return None, DictArgument(
                type="dict",
                value={
                    k: pack_result(session_id, key + [k], v, recurse=False)[1]
                    for k, v in obj.items()
                },
            )
        else:
            return None, UnsupportedArgument(type="unsupported", value=None)
    elif isinstance(obj, tuple):
        if recurse:
            return None, TupleArgument(
                type="tuple",
                value=tuple(
                    pack_result(session_id, key + [i], item, recurse=False)[1]
                    for i, item in enumerate(obj)
                ),
            )
        else:
            return None, UnsupportedArgument(type="unsupported", value=None)
    elif isinstance(obj, set):
        if recurse:
            return None, SetArgument(
                type="set",
                value={
                    pack_result(session_id, key + [i], item, recurse=False)[1]
                    for i, item in enumerate(obj)
                },
            )
        else:
            return None, UnsupportedArgument(type="unsupported", value=None)
    elif isinstance(obj, HorseData):
        if not recurse:
            node = create_obj_node(session_id, obj)
            return None, NodeArgument(type="node", value=node.id)

        result_node = create_obj_node(session_id, obj)
        result_dict = {}
        for attr_name, attr_value in obj.__dict__.items():
            if isinstance(obj, HorseVariable) and attr_name in (
                "predecessors",
                "name",
                "grad_fn",
            ):
                continue

            else:
                result_dict[attr_name] = pack_result(
                    session_id, key + [attr_name], attr_value, recurse=False
                )[1]

        return result_dict, NodeArgument(type="node", value=result_node.id)

    else:
        return None, UnsupportedArgument(type="unsupported", value=None)


@router.post(
    "/sessions/{session_id}/resources/{module_name}/{function_name}",
    response_model=ResourceResponse,
)
async def post_resource(session_id, module_name, function_name, body: dict = Body(...)):
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

    kwargs = {
        key: create_argument(type=arg["type"], value=arg["value"])
        for key, arg in body.items()
    }

    await keep_alive(session_id)

    if "." in function_name:
        class_name, function_name = function_name.split(".")
    else:
        class_name = None

    processed_kwargs = {}
    errors = []
    for key, arg in kwargs.items():
        try:
            processed_kwargs[key] = unpack_argument(session_id, [key], arg)
        except InvalidArgumentException as e:
            errors.append(e.message)

    if errors:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=errors)

    result = await execute(module_name, class_name, function_name, processed_kwargs)

    result_data, result_argument = pack_result(session_id, [], result, recurse=True)

    return ResourceResponse(
        result=result_argument,
        data=result_data,
    )
