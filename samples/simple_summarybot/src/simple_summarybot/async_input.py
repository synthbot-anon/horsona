import sys

import aiofiles


async def async_input(prompt: str) -> str:
    sys.stdout.write(prompt)
    sys.stdout.flush()

    async with aiofiles.open("/dev/stdin", mode="r") as file:
        contents = await file.readline()
    return contents
