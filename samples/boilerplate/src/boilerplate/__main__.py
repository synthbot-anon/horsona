import asyncio

from dotenv import load_dotenv

from horsona.config import load_indices, load_llms

from .async_input import async_input

load_dotenv()

llms = load_llms()
indices = load_indices()


async def main():
    reasoning_llm = llms["reasoning_llm"]

    while True:
        user_input = await async_input("Input: ")
        llm_response = await reasoning_llm.query_block("text", TASK=user_input)

        print(llm_response)


asyncio.run(main())
