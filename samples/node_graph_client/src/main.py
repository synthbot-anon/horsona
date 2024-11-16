import asyncio

from horsona_modules_client.api.default import (
    new_method_api_sessions_session_id_resources_horsona_llm_cerebras_engine_async_cerebras_engine_query_block_post,
    new_method_api_sessions_session_id_resources_horsona_llm_get_llm_engine_post,
)
from horsona_modules_client.models.args_async_chat_engine_query_block import (
    ArgsAsyncChatEngineQueryBlock,
)
from horsona_modules_client.models.args_get_llm_engine import ArgsGetLlmEngine
from horsona_modules_client.models.str_argument import StrArgument
from horsona_node_graph_client import Client
from horsona_node_graph_client.api.default import create_session_api_sessions_post


async def main():
    client = Client(base_url="http://localhost:8000")

    session_response = await create_session_api_sessions_post.asyncio(client=client)
    session_id = session_response.session_id

    reasoning_llm = await new_method_api_sessions_session_id_resources_horsona_llm_get_llm_engine_post.asyncio(
        client=client,
        session_id=session_id,
        body=ArgsGetLlmEngine(name=StrArgument(value="reasoning_llm")),
    )

    hello_task = ArgsAsyncChatEngineQueryBlock(
        self_=reasoning_llm.result,
        block_type=StrArgument(value="text"),
    )
    hello_task.additional_properties["TASK"] = StrArgument(
        value="Say hello world"
    ).to_dict()

    hello_response = await new_method_api_sessions_session_id_resources_horsona_llm_cerebras_engine_async_cerebras_engine_query_block_post.asyncio(
        client=client,
        session_id=session_id,
        body=hello_task,
    )

    print(hello_response.result)


asyncio.run(main())
