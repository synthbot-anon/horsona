import asyncio

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from horsona.interface import node_graph


async def main():
    load_dotenv(".env")

    app = FastAPI()
    app.include_router(node_graph.api_router)
    node_graph.configure(extra_modules=["node_graph_api.exposed_module"])

    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
