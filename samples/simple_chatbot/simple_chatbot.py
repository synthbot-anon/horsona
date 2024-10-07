import asyncio
import json
import sys

import aiofiles
from dotenv import load_dotenv
from horsona.autodiff.basic import step
from horsona.autodiff.variables import Value
from horsona.llm import engines_from_config
from horsona.stories.reader import StoryReaderModule

# Load API keys from .env file
load_dotenv()

# Load the reasoning_llm engine
with open("llm_config.json") as f:
    config = json.load(f)
engines = engines_from_config(config)
reasoning_llm = engines["reasoning_llm"]

# Load the chatbot config
with open("config.json") as f:
    character_info = json.load(f)


async def async_input(prompt: str) -> str:
    sys.stdout.write(prompt)
    sys.stdout.flush()

    async with aiofiles.open("/dev/stdin", mode="r") as file:
        contents = await file.readline()
    return contents


async def main():
    reader = StoryReaderModule(reasoning_llm)
    context = ""

    user_msg = await async_input("User: ")

    while True:
        user_msg = user_msg.strip()
        if not user_msg:
            break

        context, loss1 = await reader.read(Value("The user says: " + user_msg))

        response = await reasoning_llm.query_block(
            "text",
            CONTEXT=context,
            USER_MESSAGE=user_msg,
            CHARACTER_NAME=character_info["name"],
            CHARACTER_INFO=character_info,
            TASK=(
                f"You are CHARACTER_NAME, a character in a story. "
                "You are currently in a conversation with the USER. "
                "Provide a response from CHARACTER_NAME to the USER_MESSAGE."
            ),
        )

        print(f"{character_info['name']}: {response}")

        async def update():
            context, loss2 = await reader.read(
                Value(f"{character_info['name']} says: " + response)
            )
            gradients = await (loss1 + loss2).backward(reader.parameters())
            await step(gradients)

        user_msg, _ = await asyncio.gather(async_input("User: "), update())


asyncio.run(main())
