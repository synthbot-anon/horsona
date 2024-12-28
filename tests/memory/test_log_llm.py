import pytest
from pydantic import BaseModel

from horsona.autodiff.variables import Value
from horsona.memory.log_llm import LogLLMEngine
from horsona.memory.log_module import LogModule

STORY_PARAGRAPHS = """
Mr. Peterson had flown to Hofvarpnir’s offices in Berlin. He had introduced himself as “a vice president,” and proceeded to give a dry presentation on Hasbro’s current strategy. “Toy sales aren’t flat, but our stock isn’t going to double again like it did in the early 2000s if we only sell plastic to children. We have to adapt to the market and that means video games and IP licenses. Our previous forays into video games have been a bit disappointing, so we’re going to license the IP to people who have a track record for excellence,” he said, with contentless slides in the background.
Lars, the head of business development, sat in the conference room, dreaming of all of Hasbro’s juicy and profitable intellectual properties that they could license. Big brands like G.I. Joe and Transformers! Heck, if they also started making games based off movies based off board games like Battleship and Monopoly...
“So we want to license the My Little Pony franchise to you. It’s one of our most trending brands, and I think you guys could do a lot with it,” said Mr. Peterson. Lars turned to Hanna, the CEO, who looked intrigued. Mr. Peterson grinned. Silence fell across the conference room.
“Are you fucking serious?” Lars said, breaking the silence after he realized Hanna wasn’t going to jump in. “You see that statue in the corner?” He pointed to a a seven foot tall resin model of a blond muscular man. The man’s hair was wild, and his eyes glowed ice blue. He was wielding a giant battleaxe covered with dried blood and wore a wolfskin over his head. “That’s Vali. He’s one of the major characters in The Fall of Asgard, our ESRB M rated, PEGI 18+ rated, banned in Australia video game.” Lars crossed his arms, as if just pointing the man at the statue of Vali was a sufficient rebuke.
“I want to hear what Mr. Peterson has to say,” said Hanna. She flipped an unlit cigarette between her fingers. “Tell us, Mr. Peterson, why My Little Pony?”
""".strip().split("\n")


@pytest.fixture
async def log_module(reasoning_llm):
    result = LogModule(reasoning_llm)
    for paragraph in STORY_PARAGRAPHS:
        await result.append(Value("Story paragraph", paragraph))
    return result


@pytest.fixture
async def log_llm(reasoning_llm, log_module):
    return LogLLMEngine(reasoning_llm, log_module)


@pytest.mark.asyncio
async def test_query_with_story_context(log_llm, log_module):
    response = await log_llm.query_block(
        "text",
        TASK="Where did Mr. Peterson fly to?",
    )

    assert "berlin" in response.lower() or "hofvarpnir" in response.lower()


@pytest.mark.asyncio
async def test_query_object_with_story_context(log_llm, log_module):
    class Response(BaseModel):
        department: str

    # Query should return structured response referencing conversation
    response = await log_llm.query_object(
        Response,
        TASK="What is Lars the head of?",
    )

    assert response.department.lower() == "business development"
