import asyncio
import json

import pytest
from fastapi import FastAPI, Response, status
from fastapi.testclient import TestClient

from horsona.autodiff.basic import HorseVariable
from horsona.autodiff.losses import apply_loss
from horsona.autodiff.variables import Value
from horsona.character.dialogue import DialogueModule
from horsona.interface import node_graph
from horsona.interface.node_graph.node_graph_api import Argument, ArgumentType
from horsona.interface.node_graph.node_graph_models import (
    Argument,
    CreateSessionResponse,
    DictArgument,
    FloatArgument,
    IntArgument,
    ListArgument,
    ListResourcesResponse,
    NodeArgument,
    ResourceResponse,
    SchemaArgument,
    StrArgument,
    create_argument,
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
    from horsona.config import get_llm

    node_graph.configure()

    # Create a session
    create_session_response: Response = client.post("/api/sessions")
    assert create_session_response.status_code == status.HTTP_200_OK
    create_session_obj = CreateSessionResponse(**create_session_response.json())
    session_id = create_session_obj.session_id

    # Create a Value that wraps a float
    create_float_value_response: Response = client.post(
        f"/api/sessions/{session_id}/resources/{Value.__module__}/{Value.__name__}.__init__",
        json={
            "datatype": StrArgument(value="Some number").model_dump(),
            "value": FloatArgument(value=1.0).model_dump(),
        },
    )
    assert create_float_value_response.status_code == status.HTTP_200_OK
    create_float_value_obj = ResourceResponse(**create_float_value_response.json())

    assert create_float_value_obj.data["datatype"] == StrArgument(value="Some number")
    assert create_float_value_obj.data["value"] == FloatArgument(value=1.0)

    # Create a Value that wraps another Value
    create_value_value_response: Response = client.post(
        f"/api/sessions/{session_id}/resources/{Value.__module__}/{Value.__name__}.__init__",
        json={
            "datatype": StrArgument(value="Some number").model_dump(),
            "value": create_float_value_obj.result.model_dump(),
        },
    )
    assert create_value_value_response.status_code == status.HTTP_200_OK
    create_value_value_obj = ResourceResponse(**create_value_value_response.json())

    assert create_value_value_obj.data["datatype"] == StrArgument(value="Some number")
    assert create_value_value_obj.data["value"] == create_float_value_obj.result

    # Create an LLM engine
    create_llm_response: Response = client.post(
        f"/api/sessions/{session_id}/resources/{get_llm.__module__}/{get_llm.__name__}",
        json={"name": StrArgument(value="reasoning_llm").model_dump()},
    )
    assert create_llm_response.status_code == status.HTTP_200_OK
    create_llm_obj = ResourceResponse(**create_llm_response.json())
    assert isinstance(create_llm_obj.result, NodeArgument)

    derive_value_response: Response = client.post(
        f"/api/sessions/{session_id}/resources/{Value.__module__}/{Value.__name__}.derive",
        json={
            "self": create_float_value_obj.result.model_dump(),
            "value": FloatArgument(value=2.0).model_dump(),
        },
    )
    assert derive_value_response.status_code == status.HTTP_200_OK
    derive_value_obj = ResourceResponse(**derive_value_response.json())
    assert derive_value_obj.data["datatype"] == StrArgument(value="Some number")
    assert derive_value_obj.data["value"] == FloatArgument(value=2.0)


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
        f"/api/sessions/{session_id}/resources/invalid_module/invalid_function",
        json={},
    )

    assert create_invalid_module_response.status_code == status.HTTP_404_NOT_FOUND
    assert "Module not found" in create_invalid_module_response.json()["detail"]


@pytest.mark.asyncio
@pytest.mark.xdist_group(name="node_graph_sequential")
async def test_allowed_modules(client):
    from horsona.config import get_llm

    node_graph.configure()

    # Create a session
    create_session_response: Response = client.post("/api/sessions")
    assert create_session_response.status_code == status.HTTP_200_OK
    create_session_obj = CreateSessionResponse(**create_session_response.json())
    session_id = create_session_obj.session_id

    # Test that horsona module is allowed by default
    create_value_response: Response = client.post(
        f"/api/sessions/{session_id}/resources/{Value.__module__}/{Value.__name__}.__init__",
        json={
            "datatype": StrArgument(value="Test").model_dump(),
            "value": FloatArgument(value=1.0).model_dump(),
        },
    )
    assert create_value_response.status_code == status.HTTP_200_OK
    create_value_obj = ResourceResponse(**create_value_response.json())
    assert "error" not in create_value_obj.model_dump()

    # Test that other modules are disallowed by default
    create_json_response: Response = client.post(
        f"/api/sessions/{session_id}/resources/json/dumps",
        json={
            "obj": create_argument(
                type=ArgumentType.DICT,
                value={"key": StrArgument(value="value").model_dump()},
            ).model_dump(),
            "indent": IntArgument(value=2).model_dump(),
        },
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
        f"/api/sessions/{custom_session_id}/resources/{get_llm.__module__}/{get_llm.__name__}",
        json={"name": StrArgument(value="reasoning_llm").model_dump()},
    )
    assert create_llm_response.status_code == status.HTTP_200_OK

    # Test that custom modules are now allowed
    create_json_response = client.post(
        f"/api/sessions/{custom_session_id}/resources/json/dumps",
        json={
            "obj": create_argument(
                type=ArgumentType.DICT,
                value={"key": StrArgument(value="value").model_dump()},
            ).model_dump(),
            "indent": IntArgument(value=2).model_dump(),
        },
    )
    assert create_json_response.status_code == status.HTTP_200_OK

    create_random_response = client.post(
        f"/api/sessions/{custom_session_id}/resources/random/randint",
        json={
            "a": IntArgument(value=1).model_dump(),
            "b": IntArgument(value=10).model_dump(),
        },
    )
    assert create_random_response.status_code == status.HTTP_200_OK

    # Test that non-specified modules are still disallowed
    create_os_response: Response = client.post(
        f"/api/sessions/{custom_session_id}/resources/os/getcwd",
        json={},
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
        f"/api/sessions/{session_id}/resources/horsona.autodiff.variables/Value.__init__",
        json={
            "datatype": StrArgument(value="test").model_dump(),
            "value": FloatArgument(value=1.0).model_dump(),
        },
    )
    assert create_value_response.status_code == status.HTTP_200_OK

    # Wait for the session to timeout
    await asyncio.sleep(0.8)

    # Attempt to use the timed-out session
    create_timed_out_response: Response = client.post(
        f"/api/sessions/{session_id}/resources/{Value.__module__}/{Value.__name__}.__init__",
        json={
            "datatype": StrArgument(value="test").model_dump(),
            "value": FloatArgument(value=2.0).model_dump(),
        },
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
        f"/api/sessions/{new_session_id}/resources/{Value.__module__}/{Value.__name__}.__init__",
        json={
            "datatype": StrArgument(value="test").model_dump(),
            "value": FloatArgument(value=3.0).model_dump(),
        },
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
        f"/api/sessions/{session_id}/resources/{Value.__module__}/{Value.__name__}.__init__",
        json={
            "datatype": StrArgument(value="Test Value").model_dump(),
            "value": FloatArgument(value=42.0).model_dump(),
        },
    )
    assert create_value_response.status_code == status.HTTP_200_OK
    create_value_obj = ResourceResponse(**create_value_response.json())

    # Verify the created Value object
    assert create_value_obj.data["datatype"] == StrArgument(value="Test Value")
    assert create_value_obj.data["value"] == FloatArgument(value=42.0)

    # List resources
    list_resources_response: Response = client.get(
        f"/api/sessions/{session_id}/resources"
    )
    assert list_resources_response.status_code == status.HTTP_200_OK
    list_resources_obj = ListResourcesResponse(**list_resources_response.json())

    # Verify that the created Value object is in the list of resources
    assert len(list_resources_obj.resources) == 1
    assert list_resources_obj.resources[0] == create_value_obj


@pytest.mark.asyncio
@pytest.mark.xdist_group(name="node_graph_sequential")
async def test_openapi(client):
    node_graph.configure()

    openapi_response: Response = client.get("/api/openapi.json")
    assert openapi_response.status_code == status.HTTP_200_OK


@pytest.mark.asyncio
@pytest.mark.xdist_group(name="node_graph_sequential")
async def test_skipped_functions(client):
    node_graph.configure()

    openapi_response: Response = client.get("/api/openapi.json")
    assert (
        len(node_graph.skipped_functions) == 0
    ), f"Node Graph API skipped {len(node_graph.skipped_functions)} functions"


@pytest.mark.asyncio
@pytest.mark.xdist_group(name="node_graph_sequential")
async def test_llm_query_object(client):
    # Test backpropagation through API
    from pydantic import BaseModel

    from horsona.config import get_llm
    from horsona.llm.base_engine import AsyncLLMEngine

    node_graph.configure()

    class PonyName(BaseModel):
        name: str

    # Create a session
    create_session_response: Response = client.post("/api/sessions")
    assert create_session_response.status_code == status.HTTP_200_OK
    create_session_obj = CreateSessionResponse(**create_session_response.json())
    session_id = create_session_obj.session_id

    # Create an LLM engine
    create_llm_response: Response = client.post(
        f"/api/sessions/{session_id}/resources/{get_llm.__module__}/{get_llm.__name__}",
        json={"name": StrArgument(value="reasoning_llm").model_dump()},
    )
    assert create_llm_response.status_code == status.HTTP_200_OK
    create_llm_obj = ResourceResponse(**create_llm_response.json())

    # Create input text Value
    create_text_response: Response = client.post(
        f"/api/sessions/{session_id}/resources/{AsyncLLMEngine.__module__}/{AsyncLLMEngine.__name__}.{AsyncLLMEngine.query_object.__name__}",
        json={
            "self": create_llm_obj.result.model_dump(),
            "response_model": SchemaArgument(
                value=json.dumps(PonyName.model_json_schema())
            ).model_dump(),
            "TASK": StrArgument(value="Give me Twilight Sparkle's name.").model_dump(),
        },
    )
    assert create_text_response.status_code == status.HTTP_200_OK
    response_obj = ResourceResponse(**create_text_response.json())

    assert response_obj.result.value["name"] == StrArgument(value="Twilight Sparkle")


@pytest.mark.asyncio
@pytest.mark.xdist_group(name="node_graph_sequential")
async def test_llm_query_object(client):
    # Test backpropagation through API
    from pydantic import BaseModel

    from horsona.config import get_llm
    from horsona.llm.base_engine import AsyncLLMEngine

    node_graph.configure()

    class PonyName(BaseModel):
        name: str

    # Create a session
    create_session_response: Response = client.post("/api/sessions")
    assert create_session_response.status_code == status.HTTP_200_OK
    create_session_obj = CreateSessionResponse(**create_session_response.json())
    session_id = create_session_obj.session_id

    # Create an LLM engine
    create_llm_response: Response = client.post(
        f"/api/sessions/{session_id}/resources/{get_llm.__module__}/{get_llm.__name__}",
        json={"name": StrArgument(value="reasoning_llm").model_dump()},
    )
    assert create_llm_response.status_code == status.HTTP_200_OK
    create_llm_obj = ResourceResponse(**create_llm_response.json())

    # Create input text Value
    create_text_response: Response = client.post(
        f"/api/sessions/{session_id}/resources/{AsyncLLMEngine.__module__}/{AsyncLLMEngine.__name__}.{AsyncLLMEngine.query_object.__name__}",
        json={
            "self": create_llm_obj.result.model_dump(),
            "response_model": SchemaArgument(
                value=json.dumps(PonyName.model_json_schema())
            ).model_dump(),
            "TASK": StrArgument(value="Give me Twilight Sparkle's name.").model_dump(),
        },
    )
    assert create_text_response.status_code == status.HTTP_200_OK
    response_obj = ResourceResponse(**create_text_response.json())

    assert response_obj.result.value["name"] == StrArgument(value="Twilight Sparkle")


@pytest.mark.asyncio
@pytest.mark.xdist_group(name="node_graph_sequential")
async def test_json_schema(client):
    # Test backpropagation through API
    from pydantic import BaseModel

    from horsona.config import get_llm
    from horsona.llm.base_engine import AsyncLLMEngine

    node_graph.configure()

    # Create a session
    create_session_response: Response = client.post("/api/sessions")
    assert create_session_response.status_code == status.HTTP_200_OK
    create_session_obj = CreateSessionResponse(**create_session_response.json())
    session_id = create_session_obj.session_id

    # Create an LLM engine
    create_llm_response: Response = client.post(
        f"/api/sessions/{session_id}/resources/{get_llm.__module__}/{get_llm.__name__}",
        json={"name": StrArgument(value="reasoning_llm").model_dump()},
    )
    assert create_llm_response.status_code == status.HTTP_200_OK
    create_llm_obj = ResourceResponse(**create_llm_response.json())

    # Create input text Value
    create_text_response: Response = client.post(
        f"/api/sessions/{session_id}/resources/{AsyncLLMEngine.__module__}/{AsyncLLMEngine.__name__}.{AsyncLLMEngine.query_object.__name__}",
        json={
            "self": create_llm_obj.result.model_dump(),
            "response_model": SchemaArgument(
                value="""{
  "type": "object",
  "properties": {
    "pose": {
      "type": [
        "string",
        "null"
      ]
    },
    "facial_expression": {
      "type": [
        "string",
        "null"
      ]
    },
    "body_language": {
      "type": [
        "string",
        "null"
      ]
    }
  },
  "required": [
    "pose",
    "facial_expression",
    "body_language"
  ]
}"""
            ).model_dump(),
            "TASK": StrArgument(value="Give me Twilight Sparkle's name.").model_dump(),
        },
    )
    assert create_text_response.status_code == status.HTTP_200_OK
    response_obj = ResourceResponse(**create_text_response.json())


@pytest.mark.asyncio
@pytest.mark.xdist_group(name="node_graph_sequential")
async def test_backpropagation(client):
    # Test backpropagation through API
    from pydantic import BaseModel

    from horsona.autodiff.basic import HorseVariable
    from horsona.autodiff.functions import extract_object
    from horsona.autodiff.losses import apply_loss
    from horsona.config import get_llm

    node_graph.configure()

    class PonyName(BaseModel):
        name: str

    # Create a session
    create_session_response: Response = client.post("/api/sessions")
    assert create_session_response.status_code == status.HTTP_200_OK
    create_session_obj = CreateSessionResponse(**create_session_response.json())
    session_id = create_session_obj.session_id

    # Create an LLM engine
    create_llm_response: Response = client.post(
        f"/api/sessions/{session_id}/resources/{get_llm.__module__}/{get_llm.__name__}",
        json={"name": StrArgument(value="reasoning_llm").model_dump()},
    )
    assert create_llm_response.status_code == status.HTTP_200_OK
    create_llm_obj = ResourceResponse(**create_llm_response.json())

    # Create input text Value
    create_text_response: Response = client.post(
        f"/api/sessions/{session_id}/resources/{Value.__module__}/{Value.__name__}.__init__",
        json={
            "datatype": StrArgument(value="Story dialogue").model_dump(),
            "value": StrArgument(value="Hello Luna.").model_dump(),
            "llm": create_llm_obj.result.model_dump(),
        },
    )
    assert create_text_response.status_code == status.HTTP_200_OK
    create_text_obj = ResourceResponse(**create_text_response.json())

    # Get the pony's name
    extract_name_response: Response = client.post(
        f"/api/sessions/{session_id}/resources/{extract_object.__module__}/{extract_object.__name__}",
        json={
            "llm": create_llm_obj.result.model_dump(),
            "model_cls": SchemaArgument(
                value=PonyName.model_json_schema()
            ).model_dump(),
            "TEXT": create_text_obj.result.model_dump(),
            "TASK": StrArgument(value="Extract the name from the TEXT.").model_dump(),
        },
    )
    assert (
        extract_name_response.status_code == status.HTTP_200_OK
    ), extract_name_response.json()
    extract_name_obj = ResourceResponse(**extract_name_response.json())

    # Create first loss
    loss1_response: Response = client.post(
        f"/api/sessions/{session_id}/resources/{apply_loss.__module__}/{apply_loss.__name__}",
        json={
            "arg": extract_name_obj.result.model_dump(),
            "loss": StrArgument(
                value="The name should have been Celestia"
            ).model_dump(),
        },
    )
    assert loss1_response.status_code == status.HTTP_200_OK
    loss1_obj = ResourceResponse(**loss1_response.json())

    # Create second loss
    loss2_response: Response = client.post(
        f"/api/sessions/{session_id}/resources/{apply_loss.__module__}/{apply_loss.__name__}",
        json={
            "arg": extract_name_obj.result.model_dump(),
            "loss": StrArgument(
                value="They should have been addressed as Princess [...]",
            ).model_dump(),
        },
    )
    assert loss2_response.status_code == status.HTTP_200_OK
    loss2_obj = ResourceResponse(**loss2_response.json())

    # Add losses
    add_loss_response: Response = client.post(
        f"/api/sessions/{session_id}/resources/{HorseVariable.__module__}/{HorseVariable.__name__}.__add__",
        json={
            "self": loss1_obj.result.model_dump(),
            "other": loss2_obj.result.model_dump(),
        },
    )
    assert add_loss_response.status_code == status.HTTP_200_OK
    add_loss_obj = ResourceResponse(**add_loss_response.json())

    # Apply backpropagation
    step_response: Response = client.post(
        f"/api/sessions/{session_id}/resources/{HorseVariable.__module__}/{HorseVariable.__name__}.step",
        json={
            "self": add_loss_obj.result.model_dump(),
            "params": ListArgument(
                value=[create_text_obj.result.model_dump()],
            ).model_dump(),
        },
    )
    assert step_response.status_code == status.HTTP_200_OK

    # Verify the text was updated
    get_text_response: Response = client.get(
        f"/api/sessions/{session_id}/resources/{create_text_obj.result.value}"
    )
    assert get_text_response.status_code == status.HTTP_200_OK
    get_text_obj = ResourceResponse(**get_text_response.json())
    assert get_text_obj.data["value"] == StrArgument(value="Hello Princess Celestia.")


@pytest.mark.asyncio
@pytest.mark.xdist_group(name="node_graph_sequential")
async def test_dialogue_module(client):
    # Test dialogue module through API
    from pydantic import BaseModel

    from horsona.autodiff.basic import HorseVariable
    from horsona.autodiff.losses import apply_loss
    from horsona.character.dialogue import DialogueModule, DialogueResponse
    from horsona.config import get_llm

    node_graph.configure()

    # Create a session
    create_session_response: Response = client.post("/api/sessions")
    assert create_session_response.status_code == status.HTTP_200_OK
    create_session_obj = CreateSessionResponse(**create_session_response.json())
    session_id = create_session_obj.session_id

    # Create an LLM engine
    create_llm_response: Response = client.post(
        f"/api/sessions/{session_id}/resources/{get_llm.__module__}/{get_llm.__name__}",
        json={"name": StrArgument(value="reasoning_llm").model_dump()},
    )
    assert create_llm_response.status_code == status.HTTP_200_OK
    create_llm_obj = ResourceResponse(**create_llm_response.json())

    # Create character sheet Value
    character_sheet_data = {
        "name": {"type": "str", "value": "Twilight Sparkle"},
        "occupation": {"type": "str", "value": "Princess of Friendship"},
        "personality": {"type": "str", "value": "Intelligent, organized, and studious"},
        "background": {
            "type": "str",
            "value": "Student and protege of Princess Celestia",
        },
        "species": {"type": "str", "value": "Alicorn"},
    }

    create_character_sheet_response: Response = client.post(
        f"/api/sessions/{session_id}/resources/{Value.__module__}/{Value.__name__}.__init__",
        json={
            "datatype": StrArgument(value="Character Sheet").model_dump(),
            "value": DictArgument(value=character_sheet_data).model_dump(),
            "llm": create_llm_obj.result.model_dump(),
        },
    )
    assert create_character_sheet_response.status_code == status.HTTP_200_OK
    create_character_sheet_obj = ResourceResponse(
        **create_character_sheet_response.json()
    )

    # Create context Value
    create_context_response: Response = client.post(
        f"/api/sessions/{session_id}/resources/{Value.__module__}/{Value.__name__}.__init__",
        json={
            "datatype": StrArgument(value="Story context").model_dump(),
            "value": StrArgument(
                value="Twilight Sparkle has just discovered an ancient magical artifact in the Everfree Forest."
            ).model_dump(),
            "llm": create_llm_obj.result.model_dump(),
        },
    )
    assert create_context_response.status_code == status.HTTP_200_OK
    create_context_obj = ResourceResponse(**create_context_response.json())

    # Create DialogueModule
    create_dialogue_module_response: Response = client.post(
        f"/api/sessions/{session_id}/resources/{DialogueModule.__module__}/{DialogueModule.__name__}.__init__",
        json={
            "llm": create_llm_obj.result.model_dump(),
        },
    )
    assert create_dialogue_module_response.status_code == status.HTTP_200_OK
    create_dialogue_module_obj = ResourceResponse(
        **create_dialogue_module_response.json()
    )

    # Generate dialogue
    generate_dialogue_response: Response = client.post(
        f"/api/sessions/{session_id}/resources/{DialogueModule.__module__}/{DialogueModule.__name__}.generate_dialogue",
        json={
            "self": create_dialogue_module_obj.result.model_dump(),
            "character_sheet": create_character_sheet_obj.result.model_dump(),
            "context": create_context_obj.result.model_dump(),
        },
    )
    assert generate_dialogue_response.status_code == status.HTTP_200_OK
    generate_dialogue_obj = ResourceResponse(**generate_dialogue_response.json())

    # Verify dialogue value
    get_dialogue_response: Response = client.get(
        f"/api/sessions/{session_id}/resources/{generate_dialogue_obj.result.value}"
    )
    assert get_dialogue_response.status_code == status.HTTP_200_OK
    get_dialogue_obj = ResourceResponse(**get_dialogue_response.json())

    # Assert dialogue properties
    print("dialogue:", get_dialogue_obj.data)
    assert "dialogue" in get_dialogue_obj.data
    assert get_dialogue_obj.data["dialogue"].value != ""
    assert "tone" in get_dialogue_obj.data
    assert get_dialogue_obj.data["tone"].value != ""
    assert "subtext" in get_dialogue_obj.data
    assert get_dialogue_obj.data["subtext"].value != ""

    # Apply context loss
    context_loss_response: Response = client.post(
        f"/api/sessions/{session_id}/resources/{apply_loss.__module__}/{apply_loss.__name__}",
        json={
            "arg": generate_dialogue_obj.result.model_dump(),
            "loss": StrArgument(
                value="The artifact was found in the Tenochtitlan Basin, not the Everfree Forest."
            ).model_dump(),
        },
    )
    assert context_loss_response.status_code == status.HTTP_200_OK
    context_loss_obj = ResourceResponse(**context_loss_response.json())

    # Apply character loss
    character_loss_response: Response = client.post(
        f"/api/sessions/{session_id}/resources/{apply_loss.__module__}/{apply_loss.__name__}",
        json={
            "arg": generate_dialogue_obj.result.model_dump(),
            "loss": StrArgument(
                value="Twilight is a unicorn, not an alicorn."
            ).model_dump(),
        },
    )
    assert character_loss_response.status_code == status.HTTP_200_OK
    character_loss_obj = ResourceResponse(**character_loss_response.json())

    # Add losses
    add_loss_response: Response = client.post(
        f"/api/sessions/{session_id}/resources/{HorseVariable.__module__}/{HorseVariable.__name__}.__add__",
        json={
            "self": context_loss_obj.result.model_dump(),
            "other": character_loss_obj.result.model_dump(),
        },
    )
    assert add_loss_response.status_code == status.HTTP_200_OK
    add_loss_obj = ResourceResponse(**add_loss_response.json())

    # Apply backpropagation
    step_response: Response = client.post(
        f"/api/sessions/{session_id}/resources/{HorseVariable.__module__}/{HorseVariable.__name__}.step",
        json={
            "self": add_loss_obj.result.model_dump(),
            "params": ListArgument(
                value=[
                    create_context_obj.result.model_dump(),
                    create_character_sheet_obj.result.model_dump(),
                ],
            ).model_dump(),
        },
    )
    assert step_response.status_code == status.HTTP_200_OK

    # Verify context was updated
    get_context_response: Response = client.get(
        f"/api/sessions/{session_id}/resources/{create_context_obj.result.value}"
    )
    assert get_context_response.status_code == status.HTTP_200_OK
    get_context_obj = ResourceResponse(**get_context_response.json())
    assert "tenochtitlan" in get_context_obj.data["value"].value.lower()

    # Verify character sheet was updated
    get_character_sheet_response: Response = client.get(
        f"/api/sessions/{session_id}/resources/{create_character_sheet_obj.result.value}"
    )
    assert get_character_sheet_response.status_code == status.HTTP_200_OK
    get_character_sheet_obj = ResourceResponse(**get_character_sheet_response.json())
    assert "unicorn" in get_character_sheet_obj.data["value"].value["species"].lower()
