import asyncio

import uvicorn
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from horsona.interface.node_graph import NodeGraphAPI


async def main():
    load_dotenv(".env")

    node_graph_api = NodeGraphAPI(extra_modules=["node_graph_api.exposed_module"])
    await node_graph_api.start()

    # Add CORS middleware
    node_graph_api.app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allows all origins
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods
        allow_headers=["*"],  # Allows all headers
    )

    config = uvicorn.Config(
        node_graph_api.app, host="0.0.0.0", port=8000, log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
