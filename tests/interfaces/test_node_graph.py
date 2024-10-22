import os

import pytest
from dotenv import load_dotenv
from fastapi import HTTPException
from horsona.interface.node_graph.node_graph_api import Argument, NodeGraphAPI

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
        error_result = await node_graph_api.post_resource(
            session_id=session_id,
            module="invalid_module",
            function_name="invalid_function",
            kwargs={},
        )
        assert False
    except HTTPException as e:
        assert e.status_code == 400
        assert "error" in e.detail
