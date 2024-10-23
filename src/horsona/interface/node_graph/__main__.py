import asyncio

import uvicorn
from dotenv import load_dotenv

from .node_graph_api import NodeGraphAPI


async def main():
    load_dotenv()

    node_graph_api = NodeGraphAPI()

    config = uvicorn.Config(
        node_graph_api.app, host="0.0.0.0", port=8000, log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
