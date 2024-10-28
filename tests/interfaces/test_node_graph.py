import asyncio

import pytest
from fastapi import FastAPI, Response, status
from fastapi.testclient import TestClient
from horsona.autodiff.variables import Value
from horsona.interface import node_graph
from horsona.interface.node_graph.node_graph_api import Argument, ArgumentType
from horsona.interface.node_graph.node_graph_models import (
    Argument,
    CreateSessionResponse,
    ListResourcesResponse,
    PostResourceRequest,
    PostResourceResponse,
)
from horsona.llm.base_engine import AsyncLLMEngine


@pytest.fixture
async def client():
    app = FastAPI()
    app.include_router(node_graph.api_router)

    with TestClient(app) as client:
        yield client


@pytest.mark.asyncio
@pytest.mark.xdist_group(name="node_graph_sequential")
async def test_post_resource(client):
    from horsona.llm import get_llm_engine

    node_graph.configure()

    # Create a session
    create_session_response: Response = client.post("/api/sessions")
    assert create_session_response.status_code == status.HTTP_200_OK
    create_session_obj = CreateSessionResponse(**create_session_response.json())
    session_id = create_session_obj.session_id

    # Create a Value that wraps a float
    create_float_value_response: Response = client.post(
        f"/api/sessions/{session_id}/resources",
        json=PostResourceRequest(
            session_id=session_id,
            module_name=Value.__module__,
            class_name=Value.__name__,
            function_name="__init__",
            kwargs={
                "datatype": Argument(type=ArgumentType.STR, value="Some number"),
                "value": Argument(type=ArgumentType.FLOAT, value=1.0),
            },
        ).model_dump(),
    )
    assert create_float_value_response.status_code == status.HTTP_200_OK
    create_float_value_obj = PostResourceResponse(**create_float_value_response.json())

    assert create_float_value_obj.result["datatype"].type == "str"
    assert create_float_value_obj.result["datatype"].value == "Some number"
    assert create_float_value_obj.result["value"].type == "float"
    assert create_float_value_obj.result["value"].value == 1.0

    # Create a Value that wraps another Value
    create_value_value_response: Response = client.post(
        f"/api/sessions/{session_id}/resources",
        json=PostResourceRequest(
            session_id=session_id,
            module_name=Value.__module__,
            class_name=Value.__name__,
            function_name="__init__",
            kwargs={
                "datatype": Argument(type=ArgumentType.STR, value="Some number"),
                "value": Argument(
                    type=ArgumentType.NODE, value=create_float_value_obj.id
                ),
            },
        ).model_dump(),
    )
    assert create_value_value_response.status_code == status.HTTP_200_OK
    create_value_value_obj = PostResourceResponse(**create_value_value_response.json())

    assert create_value_value_obj.result["datatype"].type == "str"
    assert create_value_value_obj.result["datatype"].value == "Some number"
    assert create_value_value_obj.result["value"].type == "node"
    assert create_value_value_obj.result["value"].value == create_float_value_obj.id

    # Create an LLM engine
    create_llm_response: Response = client.post(
        f"/api/sessions/{session_id}/resources",
        json=PostResourceRequest(
            session_id=session_id,
            module_name=get_llm_engine.__module__,
            function_name=get_llm_engine.__name__,
            kwargs={"name": Argument(type=ArgumentType.STR, value="reasoning_llm")},
        ).model_dump(),
    )
    assert create_llm_response.status_code == status.HTTP_200_OK
    create_llm_obj = PostResourceResponse(**create_llm_response.json())
    assert create_llm_obj.id is not None


@pytest.mark.asyncio
@pytest.mark.xdist_group(name="node_graph_sequential")
async def test_invalid_module(client):
    node_graph.configure()
    # Create a session
    create_session_response = client.post("/api/sessions")
    assert create_session_response.status_code == status.HTTP_200_OK
    session_id = CreateSessionResponse(**create_session_response.json()).session_id

    # Test invalid module
    create_invalid_module_response: Response = client.post(
        f"/api/sessions/{session_id}/resources",
        json=PostResourceRequest(
            session_id=session_id,
            module_name="invalid_module",
            function_name="invalid_function",
            kwargs={},
        ).model_dump(),
    )

    assert create_invalid_module_response.status_code == status.HTTP_404_NOT_FOUND
    assert "Module not found" in create_invalid_module_response.json()["detail"]


@pytest.mark.asyncio
@pytest.mark.xdist_group(name="node_graph_sequential")
async def test_allowed_modules(client):
    from horsona.llm import get_llm_engine

    node_graph.configure()

    # Create a session
    create_session_response: Response = client.post("/api/sessions")
    assert create_session_response.status_code == status.HTTP_200_OK
    create_session_obj = CreateSessionResponse(**create_session_response.json())
    session_id = create_session_obj.session_id

    # Test that horsona module is allowed by default
    create_value_response: Response = client.post(
        f"/api/sessions/{session_id}/resources",
        json=PostResourceRequest(
            session_id=session_id,
            module_name=Value.__module__,
            class_name=Value.__name__,
            function_name="__init__",
            kwargs={
                "datatype": Argument(type=ArgumentType.STR, value="Test"),
                "value": Argument(type=ArgumentType.FLOAT, value=1.0),
            },
        ).model_dump(),
    )
    assert create_value_response.status_code == status.HTTP_200_OK
    create_value_obj = PostResourceResponse(**create_value_response.json())
    assert "error" not in create_value_obj.model_dump()

    # Test that other modules are disallowed by default
    create_json_response: Response = client.post(
        f"/api/sessions/{session_id}/resources",
        json=PostResourceRequest(
            session_id=session_id,
            module_name="json",
            function_name="dumps",
            kwargs={
                "obj": Argument(
                    type=ArgumentType.DICT,
                    value={"key": Argument(type=ArgumentType.STR, value="value")},
                ),
                "indent": Argument(type=ArgumentType.INT, value=2),
            },
        ).model_dump(),
    )

    assert create_json_response.status_code == status.HTTP_404_NOT_FOUND
    assert "Module not found" in create_json_response.json()["detail"]

    # Test with custom allowed modules
    node_graph.configure(extra_modules=["json", "random"])
    custom_session_response = client.post("/api/sessions")
    assert custom_session_response.status_code == status.HTTP_200_OK
    custom_session_obj = CreateSessionResponse(**custom_session_response.json())
    custom_session_id = custom_session_obj.session_id

    # Test that horsona module is still allowed
    create_llm_response = client.post(
        f"/api/sessions/{custom_session_id}/resources",
        json=PostResourceRequest(
            session_id=custom_session_id,
            module_name=get_llm_engine.__module__,
            function_name=get_llm_engine.__name__,
            kwargs={"name": Argument(type=ArgumentType.STR, value="reasoning_llm")},
        ).model_dump(),
    )
    assert create_llm_response.status_code == status.HTTP_200_OK

    # Test that custom modules are now allowed
    create_json_response = client.post(
        f"/api/sessions/{custom_session_id}/resources",
        json=PostResourceRequest(
            session_id=custom_session_id,
            module_name="json",
            function_name="dumps",
            kwargs={
                "obj": Argument(
                    type=ArgumentType.DICT,
                    value={"key": Argument(type=ArgumentType.STR, value="value")},
                ),
                "indent": Argument(type=ArgumentType.INT, value=2),
            },
        ).model_dump(),
    )
    assert create_json_response.status_code == status.HTTP_200_OK

    create_random_response = client.post(
        f"/api/sessions/{custom_session_id}/resources",
        json=PostResourceRequest(
            session_id=custom_session_id,
            module_name="random",
            function_name="randint",
            kwargs={
                "a": Argument(type=ArgumentType.INT, value=1),
                "b": Argument(type=ArgumentType.INT, value=10),
            },
        ).model_dump(),
    )
    assert create_random_response.status_code == status.HTTP_200_OK

    # Test that non-specified modules are still disallowed
    create_os_response: Response = client.post(
        f"/api/sessions/{custom_session_id}/resources",
        json=PostResourceRequest(
            session_id=custom_session_id,
            module_name="os",
            function_name="getcwd",
            kwargs={},
        ).model_dump(),
    )
    assert create_os_response.status_code == status.HTTP_404_NOT_FOUND
    assert "Module not found" in create_os_response.json()["detail"]


# Test session timeout and keep_alive
@pytest.mark.asyncio
@pytest.mark.xdist_group(name="node_graph_sequential")
async def test_session_timeout(client):
    # Configure node_graph with a short timeout
    node_graph.configure(session_timeout=0.5, session_cleanup_interval=0.25)

    # Create a session
    create_session_response: Response = client.post("/api/sessions")
    assert create_session_response.status_code == status.HTTP_200_OK
    create_session_obj = CreateSessionResponse(**create_session_response.json())
    session_id = create_session_obj.session_id

    # Verify the session is active by posting a resource
    create_value_response: Response = client.post(
        f"/api/sessions/{session_id}/resources",
        json=PostResourceRequest(
            session_id=session_id,
            module_name="horsona.autodiff.variables",
            class_name="Value",
            function_name="__init__",
            kwargs={
                "datatype": Argument(type=ArgumentType.STR, value="test"),
                "value": Argument(type=ArgumentType.FLOAT, value=1.0),
            },
        ).model_dump(),
    )
    assert create_value_response.status_code == status.HTTP_200_OK

    # Wait for the session to timeout
    await asyncio.sleep(0.8)

    # Attempt to use the timed-out session
    create_timed_out_response: Response = client.post(
        f"/api/sessions/{session_id}/resources",
        json=PostResourceRequest(
            session_id=session_id,
            module_name=Value.__module__,
            class_name=Value.__name__,
            function_name="__init__",
            kwargs={
                "datatype": Argument(type=ArgumentType.STR, value="test"),
                "value": Argument(type=ArgumentType.FLOAT, value=2.0),
            },
        ).model_dump(),
    )
    assert create_timed_out_response.status_code == status.HTTP_404_NOT_FOUND
    assert "Session not found" in create_timed_out_response.json()["detail"]

    # Create a new session to test keep_alive
    create_new_session_response: Response = client.post("/api/sessions")
    assert create_new_session_response.status_code == status.HTTP_200_OK
    create_new_session_obj = CreateSessionResponse(**create_new_session_response.json())
    new_session_id = create_new_session_obj.session_id

    # Keep the session alive
    for _ in range(3):
        await asyncio.sleep(0.35)
        keep_alive_response: Response = client.post(
            f"/api/sessions/{new_session_id}/keep_alive",
        )
        assert keep_alive_response.status_code == status.HTTP_200_OK

    # Verify the session is still active after keep_alive calls
    create_after_keep_alive_response: Response = client.post(
        f"/api/sessions/{new_session_id}/resources",
        json=PostResourceRequest(
            session_id=new_session_id,
            module_name=Value.__module__,
            class_name=Value.__name__,
            function_name="__init__",
            kwargs={
                "datatype": Argument(type=ArgumentType.STR, value="test"),
                "value": Argument(type=ArgumentType.FLOAT, value=3.0),
            },
        ).model_dump(),
    )
    assert create_after_keep_alive_response.status_code == status.HTTP_200_OK


@pytest.mark.asyncio
@pytest.mark.xdist_group(name="node_graph_sequential")
async def test_list_resources(client):
    node_graph.configure()

    # Create a session
    create_session_response: Response = client.post("/api/sessions")
    assert create_session_response.status_code == status.HTTP_200_OK
    create_session_obj = CreateSessionResponse(**create_session_response.json())
    session_id = create_session_obj.session_id

    # Create a Value object
    create_value_response: Response = client.post(
        f"/api/sessions/{session_id}/resources",
        json=PostResourceRequest(
            session_id=session_id,
            module_name=Value.__module__,
            class_name=Value.__name__,
            function_name="__init__",
            kwargs={
                "datatype": Argument(type=ArgumentType.STR, value="Test Value"),
                "value": Argument(type=ArgumentType.FLOAT, value=42.0),
            },
        ).model_dump(),
    )
    assert create_value_response.status_code == status.HTTP_200_OK
    create_value_obj = PostResourceResponse(**create_value_response.json())

    # Verify the created Value object
    assert create_value_obj.result["datatype"].type == ArgumentType.STR
    assert create_value_obj.result["datatype"].value == "Test Value"
    assert create_value_obj.result["value"].type == ArgumentType.FLOAT
    assert create_value_obj.result["value"].value == 42.0

    # List resources
    list_resources_response: Response = client.get(
        f"/api/sessions/{session_id}/resources"
    )
    assert list_resources_response.status_code == status.HTTP_200_OK
    list_resources_obj = ListResourcesResponse(**list_resources_response.json())

    # Verify that the created Value object is in the list of resources
    assert len(list_resources_obj.resources) == 1
    assert list_resources_obj.resources[0].id == create_value_obj.id
    assert list_resources_obj.resources[0].result == create_value_obj.result


async def extract_pony_name(llm: AsyncLLMEngine, text: Value[str]):
    from horsona.autodiff.functions import extract_object
    from pydantic import BaseModel

    class PonyName(BaseModel):
        name: str

    return await extract_object(
        llm, PonyName, TEXT=text, TASK="Extract the name from the TEXT."
    )


@pytest.mark.asyncio
@pytest.mark.xdist_group(name="node_graph_sequential")
async def test_backpropagation(client):
    # Test backpropagation through API
    from horsona.autodiff.basic import HorseVariable
    from horsona.autodiff.losses import apply_loss
    from horsona.llm import get_llm_engine

    node_graph.configure(extra_modules=[extract_pony_name.__module__])

    # Create a session
    create_session_response: Response = client.post("/api/sessions")
    assert create_session_response.status_code == status.HTTP_200_OK
    create_session_obj = CreateSessionResponse(**create_session_response.json())
    session_id = create_session_obj.session_id

    # Create an LLM engine
    create_llm_response: Response = client.post(
        f"/api/sessions/{session_id}/resources",
        json=PostResourceRequest(
            session_id=session_id,
            module_name=get_llm_engine.__module__,
            function_name=get_llm_engine.__name__,
            kwargs={"name": Argument(type=ArgumentType.STR, value="reasoning_llm")},
        ).model_dump(),
    )
    assert create_llm_response.status_code == status.HTTP_200_OK
    create_llm_obj = PostResourceResponse(**create_llm_response.json())

    # Create input text Value
    create_text_response: Response = client.post(
        f"/api/sessions/{session_id}/resources",
        json=PostResourceRequest(
            session_id=session_id,
            module_name=Value.__module__,
            class_name=Value.__name__,
            function_name="__init__",
            kwargs={
                "datatype": Argument(type=ArgumentType.STR, value="Story dialogue"),
                "value": Argument(type=ArgumentType.STR, value="Hello Luna."),
                "llm": Argument(type=ArgumentType.NODE, value=create_llm_obj.id),
            },
        ).model_dump(),
    )
    assert create_text_response.status_code == status.HTTP_200_OK
    create_text_obj = PostResourceResponse(**create_text_response.json())

    # Get the pony's name
    extract_name_response: Response = client.post(
        f"/api/sessions/{session_id}/resources",
        json=PostResourceRequest(
            session_id=session_id,
            module_name=extract_pony_name.__module__,
            function_name=extract_pony_name.__name__,
            kwargs={
                "llm": Argument(type=ArgumentType.NODE, value=create_llm_obj.id),
                "text": Argument(type=ArgumentType.NODE, value=create_text_obj.id),
            },
        ).model_dump(),
    )
    assert (
        extract_name_response.status_code == status.HTTP_200_OK
    ), extract_name_response.json()
    extract_name_obj = PostResourceResponse(**extract_name_response.json())

    # Create first loss
    loss1_response: Response = client.post(
        f"/api/sessions/{session_id}/resources",
        json=PostResourceRequest(
            session_id=session_id,
            module_name=apply_loss.__module__,
            function_name=apply_loss.__name__,
            kwargs={
                "arg": Argument(type=ArgumentType.NODE, value=extract_name_obj.id),
                "loss": Argument(
                    type=ArgumentType.STR, value="The name should have been Celestia"
                ),
            },
        ).model_dump(),
    )
    assert loss1_response.status_code == status.HTTP_200_OK
    loss1_obj = PostResourceResponse(**loss1_response.json())

    # Create second loss
    loss2_response: Response = client.post(
        f"/api/sessions/{session_id}/resources",
        json=PostResourceRequest(
            session_id=session_id,
            module_name=apply_loss.__module__,
            function_name=apply_loss.__name__,
            kwargs={
                "arg": Argument(type=ArgumentType.NODE, value=extract_name_obj.id),
                "loss": Argument(
                    type=ArgumentType.STR,
                    value="They should have been addressed as Princess",
                ),
            },
        ).model_dump(),
    )
    assert loss2_response.status_code == status.HTTP_200_OK
    loss2_obj = PostResourceResponse(**loss2_response.json())

    # Add losses
    add_loss_response: Response = client.post(
        f"/api/sessions/{session_id}/resources",
        json=PostResourceRequest(
            session_id=session_id,
            module_name=HorseVariable.__module__,
            class_name=HorseVariable.__name__,
            function_name=HorseVariable.__add__.__name__,
            kwargs={
                "self": Argument(type=ArgumentType.NODE, value=loss1_obj.id),
                "other": Argument(type=ArgumentType.NODE, value=loss2_obj.id),
            },
        ).model_dump(),
    )
    assert add_loss_response.status_code == status.HTTP_200_OK
    add_loss_obj = PostResourceResponse(**add_loss_response.json())

    # Apply backpropagation
    step_response: Response = client.post(
        f"/api/sessions/{session_id}/resources",
        json=PostResourceRequest(
            session_id=session_id,
            module_name=HorseVariable.__module__,
            class_name=HorseVariable.__name__,
            function_name=HorseVariable.step.__name__,
            kwargs={
                "self": Argument(type=ArgumentType.NODE, value=add_loss_obj.id),
                "params": Argument(
                    type=ArgumentType.LIST,
                    value=[Argument(type=ArgumentType.NODE, value=create_text_obj.id)],
                ),
            },
        ).model_dump(),
    )
    assert step_response.status_code == status.HTTP_200_OK

    # Verify the text was updated
    get_text_response: Response = client.get(
        f"/api/sessions/{session_id}/resources/{create_text_obj.id}"
    )
    assert get_text_response.status_code == status.HTTP_200_OK
    get_text_obj = PostResourceResponse(**get_text_response.json())
    assert get_text_obj.result["value"].value == "Hello Princess Celestia."
