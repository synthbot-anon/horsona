import asyncio
import logging

import pytest
import uvicorn
from fastapi import FastAPI
from openai import AsyncOpenAI

from horsona.interface import oai


@pytest.fixture(scope="function")
async def oai_server(reasoning_llm):
    app = FastAPI()
    app.include_router(oai.api_router)
    oai.add_llm_engine(reasoning_llm)

    config = uvicorn.Config(app, host="127.0.0.1", port=8001)
    server = uvicorn.Server(config)

    # Temporarily suppress logging
    uvicorn_error_level = logging.getLogger("uvicorn.error").getEffectiveLevel()
    uvicorn_access_level = logging.getLogger("uvicorn.access").getEffectiveLevel()
    uvicorn_asgi_level = logging.getLogger("uvicorn.asgi").getEffectiveLevel()
    fastapi_level = logging.getLogger("fastapi").getEffectiveLevel()

    logging.getLogger("uvicorn.error").setLevel(logging.CRITICAL)
    logging.getLogger("uvicorn.access").setLevel(logging.CRITICAL)
    logging.getLogger("uvicorn.asgi").setLevel(logging.CRITICAL)
    logging.getLogger("fastapi").setLevel(logging.CRITICAL)
    # Start the server and wait for it to be ready
    server_task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.1)

    yield

    # Stop the server and wait for it to exit
    server.should_exit = True
    while not server_task.done():
        await asyncio.sleep(0.1)

    # Restore logging
    logging.getLogger("uvicorn.error").setLevel(uvicorn_error_level)
    logging.getLogger("uvicorn.access").setLevel(uvicorn_access_level)
    logging.getLogger("uvicorn.asgi").setLevel(uvicorn_asgi_level)
    logging.getLogger("fastapi").setLevel(fastapi_level)


@pytest.mark.asyncio
async def test_chat_completion(oai_server):
    # Initialize client pointing to local server
    oai_client = AsyncOpenAI(
        base_url="http://127.0.0.1:8001/api/v1",
        api_key="not-needed",  # API key not needed for local server
    )

    # Send chat completion request
    response = await oai_client.chat.completions.create(
        model="reasoning_llm",  # Use the reasoning_llm model
        messages=[{"role": "user", "content": "say hello world"}],
    )

    # Verify response structure
    assert response.choices is not None
    assert len(response.choices) > 0
    assert response.choices[0].message is not None
    assert response.choices[0].message.content is not None
    assert isinstance(response.choices[0].message.content, str)

    assert (
        "hello" in response.choices[0].message.content.lower()
        and "world" in response.choices[0].message.content.lower()
    )


@pytest.mark.asyncio
async def test_chat_completion_stream(oai_server):
    # Initialize client pointing to local server
    oai_client = AsyncOpenAI(
        base_url="http://127.0.0.1:8001/api/v1",
        api_key="not-needed",  # API key not needed for local server
    )

    # Send streaming chat completion request
    stream = await oai_client.chat.completions.create(
        model="reasoning_llm",
        messages=[{"role": "user", "content": "say hello world"}],
        stream=True,
    )

    # Collect all chunks
    chunks = []
    content = ""
    async for chunk in stream:
        chunks.append(chunk)
        if chunk.choices[0].delta.content:
            content += chunk.choices[0].delta.content

    # Verify chunks structure
    assert len(chunks) > 0
    assert all(hasattr(chunk, "choices") for chunk in chunks)
    assert all(len(chunk.choices) > 0 for chunk in chunks)
    assert all(hasattr(chunk.choices[0], "delta") for chunk in chunks)

    # Verify final content
    assert "hello" in content.lower() and "world" in content.lower()
