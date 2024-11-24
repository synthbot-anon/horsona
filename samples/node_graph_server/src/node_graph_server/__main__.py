import argparse
import asyncio

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

from horsona.interface import node_graph

from . import exposed_module


async def main():
    load_dotenv(".env")

    parser = argparse.ArgumentParser(description="Node Graph API")
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host to run the server on"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on"
    )
    args = parser.parse_args()

    app = FastAPI(title="Horsona Node Graph")
    app.include_router(node_graph.api_router)
    node_graph.configure(extra_modules=[exposed_module.__name__], session_timeout=9e9)

    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
