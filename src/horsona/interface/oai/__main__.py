import argparse
import asyncio

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

from horsona.interface import oai
from horsona.llm import load_engines

load_dotenv()
engines = load_engines()


async def main():
    parser = argparse.ArgumentParser(description="Horsona OAI API")
    parser.add_argument(
        "--extra-modules", nargs="+", default=[], help="Extra modules to allow"
    )
    args = parser.parse_args()

    app = FastAPI(title="Horsona OAI")
    app.include_router(oai.api_router)
    for engine in engines.values():
        oai.add_llm_engine(engine)

    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
