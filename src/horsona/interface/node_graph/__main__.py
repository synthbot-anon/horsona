import argparse
import asyncio

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from horsona.interface import node_graph


async def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Node Graph API")
    parser.add_argument(
        "--session-timeout", type=float, default=300, help="Session timeout in seconds"
    )
    parser.add_argument(
        "--session-cleanup-interval",
        type=float,
        default=60,
        help="Session cleanup interval in seconds",
    )
    parser.add_argument(
        "--extra-modules", nargs="+", default=[], help="Extra modules to allow"
    )
    args = parser.parse_args()

    app = FastAPI()
    app.include_router(node_graph.api_router)
    node_graph.configure(
        session_timeout=args.session_timeout,
        session_cleanup_interval=args.session_cleanup_interval,
        extra_modules=args.extra_modules,
    )

    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
