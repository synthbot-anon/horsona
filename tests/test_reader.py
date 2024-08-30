import functools

import pytest
from pydantic import BaseModel

from horsona.autodiff import HorseFunction, HorseOptimizer, HorseVariable
from horsona.autodiff.functions import TextExtractor
from horsona.autodiff.losses import ConstantLoss
from horsona.autodiff.variables import Value
from horsona.llm.cerebras_engine import AsyncCerebrasEngine
from horsona.llm.fireworks_engine import AsyncFireworksEngine
from horsona.stories.reader import StoryReader

STORY = """James looked skeptically at his friend David as he sat down at computer #12.
David had won the Hasbro raffle for one of fifteen all-expenses-paid trips for two to Pawtucket, Rhode Island to play the first alpha build of the official My Little Pony MMO: Equestria Online. Hasbro had claimed that a game that revolved so heavily around friendship needed actual friends to test properly.
“Look, I’m glad you invited me,” James said as he picked up his head set. “And it’s cool that I get to try something before anyone else. But I’m still not sure I really want to play a game that’s so...so...pink and purple.”
David scoffed. “I know you. What was that Korean MMO with the little girls that you were so into?  You’ll be running dungeons over and over again, just like you always do with every MMO that comes out. But me?” David gave the widest smile possible. “I’m here for the ponies. We both get what we want!”"""


@pytest.fixture(scope="module")
def reasoning_llm():
    return AsyncCerebrasEngine(
        model="llama3.1-70b",
        fallback=AsyncFireworksEngine(
            model="accounts/fireworks/models/llama-v3p1-70b-instruct"
        ),
    )


@pytest.mark.asyncio
async def test_reader(reasoning_llm):
    story_paragraphs = STORY.split("\n")
    reader = StoryReader(reasoning_llm)

    optimizer = HorseOptimizer(reader.parameters())

    for p in story_paragraphs:
        # Figure out what's new in the paragraph
        loss = await reader.read(p)

        # Update the reader's state based on what was read
        await optimizer.zero_grad()
        await loss.backward()
        await optimizer.step()

    assert len(reader.short_term_memory.buffer) > 0
    assert len(reader.long_term_memory.context) > 0
    assert reader.current_state.last_speaker == "David"
    assert set(reader.current_state.characters_in_scene) == {"James", "David"}
    assert (
        "computer #12" in reader.current_state.current_location.lower()
        or "pawtucket" in reader.current_state.current_location.lower()
    )
