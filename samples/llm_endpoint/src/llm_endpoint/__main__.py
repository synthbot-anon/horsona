import asyncio
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

from horsona.autodiff.basic import unzip, zip
from horsona.autodiff.variables import Value
from horsona.config import load_indices, load_llms
from horsona.database.embedding_database import EmbeddingDatabase
from horsona.interface import oai
from horsona.memory.wiki_module import WikiModule

from .backstory_llm import BackstoryLLMEngine

load_dotenv()

llms = load_llms()
indices = load_indices()


async def load_backstory_llm() -> BackstoryLLMEngine:
    if os.path.exists("./backstory_llm_state.zip"):
        state_dict = unzip("./backstory_llm_state.zip")
        backstory_llm: BackstoryLLMEngine = BackstoryLLMEngine.load_state_dict(
            state_dict,
        )
        backstory_module = backstory_llm.backstory_module
    else:
        reasoning_llm = llms["reasoning_llm"]
        query_index = indices["query_index"]
        embedding_db = EmbeddingDatabase(reasoning_llm, query_index)

        backstory_module = WikiModule(
            reasoning_llm,
            embedding_db,
            guidelines="Make sure to include specific people, places, events, and objects mentioned.",
        )
        backstory_llm: BackstoryLLMEngine = BackstoryLLMEngine(
            reasoning_llm,
            backstory_module,
        )

    # Walk through all files in ./backstory directory
    for root, dirs, files in os.walk("./backstory"):
        for file in sorted(files):
            print(f"Adding file: {root}/{file}")
            # Get full file path
            file_path = os.path.join(root, file)

            # Read file content
            with open(file_path, "r") as f:
                content = f.read()

            # Get folder name from relative path
            folder = os.path.relpath(root, "./backstory")
            if folder == ".":
                folder = "root"

            # Add file to backstory module
            new_module = await backstory_module.add_file(
                filepath=file_path, content=Value("Backstory content", content)
            )

            if new_module:
                state_dict = backstory_llm.state_dict()
                zip(state_dict, f"./backstory_llm_state.zip.new")
                os.rename(
                    f"./backstory_llm_state.zip.new", f"./backstory_llm_state.zip"
                )

    return backstory_llm


async def main():
    app = FastAPI(title="Horsona LLM")
    app.include_router(oai.api_router)
    backstory_llm = await load_backstory_llm()

    oai.add_llm_engine(backstory_llm, name="sample-endpoint")

    config = uvicorn.Config(app, host="0.0.0.0", port=8001, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


asyncio.run(main())
