import os

import pytest
from dotenv import load_dotenv
from fastapi import HTTPException
from horsona.interface.node_graph import NodeGraphAPI
from horsona.interface.node_graph.node_graph_api import Argument

# Load environment variables from .env file
load_dotenv()


@pytest.mark.asyncio
async def test_node_graph():
    node_graph_api = NodeGraphAPI()
    session_id = (await node_graph_api.create_session())["session_id"]
    print("session id:", session_id)

    result1 = await node_graph_api.post_resource(
        session_id=session_id,
        module="horsona.autodiff.variables",
        class_name="Value",
        function_name="__init__",
        kwargs={
            "datatype": Argument(type="str", value="Some number"),
            "value": Argument(type="float", value=1.0),
        },
    )

    assert result1["result"]["datatype"].type == "str"
    assert result1["result"]["datatype"].value == "Some number"
    assert result1["result"]["value"].type == "float"
    assert result1["result"]["value"].value == 1.0

    result2 = await node_graph_api.post_resource(
        session_id=session_id,
        module="horsona.autodiff.variables",
        class_name="Value",
        function_name="__init__",
        kwargs={
            "datatype": Argument(type="str", value="Some number"),
            "value": Argument(type="node", value=result1["id"]),
        },
    )

    assert result2["result"]["datatype"].type == "str"
    assert result2["result"]["datatype"].value == "Some number"
    assert result2["result"]["value"].type == "node"
    assert result2["result"]["value"].value == result1["id"]

    llm_api_result = await node_graph_api.post_resource(
        session_id=session_id,
        module="horsona.llm",
        function_name="get_llm_engine",
        kwargs={"name": Argument(type="str", value="reasoning_llm")},
    )

    assert "error" not in llm_api_result

    try:
        await node_graph_api.post_resource(
            session_id=session_id,
            module="invalid_module",
            function_name="invalid_function",
            kwargs={},
        )
        assert False
    except HTTPException as e:
        assert e.status_code == 404


@pytest.mark.asyncio
async def test_allowed_modules():
    # Test with default allowed modules
    default_api = NodeGraphAPI()
    default_session_id = (await default_api.create_session())["session_id"]

    # Test that horsona module is allowed by default
    horsona_result = await default_api.post_resource(
        session_id=default_session_id,
        module="horsona.autodiff.variables",
        class_name="Value",
        function_name="__init__",
        kwargs={
            "datatype": Argument(type="str", value="Test"),
            "value": Argument(type="float", value=1.0),
        },
    )
    assert "error" not in horsona_result

    # Test that other modules are disallowed by default
    with pytest.raises(HTTPException) as exc_info:
        await default_api.post_resource(
            session_id=default_session_id,
            module="json",
            function_name="dumps",
            kwargs={
                "obj": Argument(type="dict", value={"key": "value"}),
                "indent": Argument(type="int", value=2),
            },
        )
    assert exc_info.value.status_code == 404
    assert "Module not found" in str(exc_info.value.detail)

    # Test with custom allowed modules
    custom_api = NodeGraphAPI(extra_modules=["json", "random"])
    custom_session_id = (await custom_api.create_session())["session_id"]

    # Test that horsona module is still allowed
    horsona_result = await custom_api.post_resource(
        session_id=custom_session_id,
        module="horsona.llm",
        function_name="get_llm_engine",
        kwargs={"name": Argument(type="str", value="reasoning_llm")},
    )
    assert "error" not in horsona_result

    # Test that custom modules are now allowed
    json_result = await custom_api.post_resource(
        session_id=custom_session_id,
        module="json",
        function_name="dumps",
        kwargs={
            "obj": Argument(type="dict", value={"key": "value"}),
            "indent": Argument(type="int", value=2),
        },
    )
    assert "error" not in json_result

    random_result = await custom_api.post_resource(
        session_id=custom_session_id,
        module="random",
        function_name="randint",
        kwargs={
            "a": Argument(type="int", value=1),
            "b": Argument(type="int", value=10),
        },
    )
    assert "error" not in random_result

    # Test that non-specified modules are still disallowed
    with pytest.raises(HTTPException) as exc_info:
        await custom_api.post_resource(
            session_id=custom_session_id,
            module="os",
            function_name="getcwd",
            kwargs={},
        )
    assert exc_info.value.status_code == 404
    assert "Module not found" in str(exc_info.value.detail)
