import asyncio
import json
import sys

import aiofiles
from dotenv import load_dotenv
from pydantic import BaseModel

from horsona.autodiff.variables import Value
from horsona.cache.db_cache import DatabaseCache
from horsona.cache.list_cache import ListCache
from horsona.database.embedding_database import EmbeddingDatabase
from horsona.index import indices_from_config
from horsona.io.reader import ReaderModule
from horsona.llm import engines_from_config

# Load API keys from .env file
load_dotenv(".env")

# Load the reasoning_llm engine
with open("llm_config.json") as f:
    config = json.load(f)
engines = engines_from_config(config)
reasoning_llm = engines["reasoning_llm"]

# Load the embedding index info
with open("index_config.json") as f:
    config = json.load(f)
indices = indices_from_config(config)
query_index = indices["query_index"]

# Load the chatbot config
with open("persona_config.json") as f:
    character_info = json.load(f)


async def async_input(prompt: str) -> str:
    sys.stdout.write(prompt)
    sys.stdout.flush()

    async with aiofiles.open("/dev/stdin", mode="r") as file:
        contents = await file.readline()
    return contents


class LiveState(BaseModel):
    current_location: str = "unknown"
    CHARACTER_mental_state: str = "unknown"
    USER_mental_state: str = "unknown"
    current_goal: str = "unknown"


async def main():
    setting_db = EmbeddingDatabase(
        reasoning_llm,
        query_index,
    )
    database_context = DatabaseCache(reasoning_llm, setting_db, 10)
    buffer_context = ListCache(5)
    state_context = Value("Live state", LiveState())

    reader = ReaderModule(reasoning_llm)
    read_context = await reader.create_context(
        database_context, buffer_context, state_context
    )

    user_msg = await async_input("User: ")

    while True:
        user_msg = user_msg.strip()
        if not user_msg:
            break

        read_context, loss1 = await reader.read(
            read_context, Value("User message", "The user says: " + user_msg)
        )

        response = await reasoning_llm.query_block(
            "text",
            CONTEXT=read_context,
            USER_MESSAGE=user_msg,
            CHARACTER_NAME=character_info["name"],
            CHARACTER_INFO=character_info,
            TASK=(
                f"You are CHARACTER_NAME, a character in a story. "
                "You are currently in a conversation with the USER. "
                "Provide a response from CHARACTER_NAME to the USER_MESSAGE."
            ),
        )

        print(f"=== {character_info['name']} ===")
        print(f"{response}")
        print()

        async def update():
            nonlocal read_context
            read_context, loss2 = await reader.read(
                read_context,
                Value(
                    "Character message", f"{character_info['name']} says: " + response
                ),
            )
            await (loss1 + loss2).step([setting_db])

        user_msg, _ = await asyncio.gather(async_input("User: "), update())


asyncio.run(main())
