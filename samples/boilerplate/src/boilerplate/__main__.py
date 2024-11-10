import asyncio

from dotenv import load_dotenv
from horsona.index import load_indices
from horsona.llm import load_engines

from .async_input import async_input

load_dotenv()

engines = load_engines()
indices = load_indices()


async def main():
    reasoning_llm = engines["reasoning_llm"]

    while True:
        user_input = await async_input("Input: ")
        llm_response = await reasoning_llm.query_block("text", TASK=user_input)

        print(llm_response)


asyncio.run(main())
