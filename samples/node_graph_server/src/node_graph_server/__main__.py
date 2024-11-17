import asyncio

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

from horsona.interface import node_graph

from . import exposed_module


async def main():
    load_dotenv(".env")

    app = FastAPI(title="Horsona Node Graph")
    app.include_router(node_graph.api_router)
    node_graph.configure(extra_modules=[exposed_module.__name__], session_timeout=9e9)

    config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
