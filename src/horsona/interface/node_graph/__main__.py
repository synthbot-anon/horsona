import asyncio

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html

from .node_graph_api import NodeGraphAPI, app


async def main():
    load_dotenv()

    node_graph_api = NodeGraphAPI()

    # Add Swagger UI route
    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        return get_swagger_ui_html(openapi_url="/openapi.json", title="API Docs")

    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
