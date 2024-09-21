import asyncio
from typing import AsyncGenerator, Optional

import pytest
from horsona.autodiff.basic import GradContext, HorseModule, horsefunction, step
from horsona.autodiff.variables import Value
from horsona.llm.base_engine import AsyncLLMEngine
from horsona.llm.cerebras_engine import AsyncCerebrasEngine
from horsona.llm.fireworks_engine import AsyncFireworksEngine
from horsona.memory.caches.cache import Cache
from horsona.stories.reader import LiveState, ReadResult, StoryReader
from pydantic import BaseModel

STORY = """James looked skeptically at his friend David as he sat down at computer #12.
David had won the Hasbro raffle for one of fifteen all-expenses-paid trips for two to Pawtucket, Rhode Island to play the first alpha build of the official My Little Pony MMO: Equestria Online. Hasbro had claimed that a game that revolved so heavily around friendship needed actual friends to test properly.
“Look, I’m glad you invited me,” James said as he picked up his head set. “And it’s cool that I get to try something before anyone else. But I’m still not sure I really want to play a game that’s so...so...pink and purple.”
David scoffed. “I know you. What was that Korean MMO with the little girls that you were so into?  You’ll be running dungeons over and over again, just like you always do with every MMO that comes out. But me?” David gave the widest smile possible. “I’m here for the ponies. We both get what we want!”"""


@pytest.mark.asyncio
async def test_reader(reasoning_llm):
    story_paragraphs = STORY.split("\n")
    reader = StoryReader(reasoning_llm)

    for p in story_paragraphs:
        # Figure out what's new in the paragraph
        loss: ReadResult = await reader.read(Value(p))

        # Update the reader's state based on what was read
        gradients = await loss.backward(reader.parameters())
        await step(gradients)

    assert len(reader.buffer_memory.context) > 0
    assert len(reader.database_memory.context) > 0
    assert reader.state_cache.context.value.last_speaker == "David"
    assert set(reader.state_cache.context.value.characters_in_scene) == {
        "James",
        "David",
    }
    assert (
        "computer #12" in reader.state_cache.context.value.current_location.lower()
        or "pawtucket" in reader.state_cache.context.value.current_location.lower()
    )


class CharacterInfo(BaseModel):
    updated_synopsis: Optional[str] = None
    updated_personality: Optional[str] = None
    updated_appearance: Optional[str] = None
    new_quotes: Optional[list[str]] = None


class CharacterCardContext(
    Cache[Value[dict[str, CharacterInfo]], Value[str | list[str]]]
):
    def __init__(self, llm: AsyncLLMEngine):
        super().__init__(Value({}))
        self.llm = llm

    async def _update_character(
        self,
        name,
        database_context: Value,
        buffer_context: Value,
        state_context: Value,
        paragraph: Value,
    ) -> None:
        new_info = {}

        if name not in self.context.value:
            current_info = new_info[name] = {
                "name": name,
                "synopsis": "unknown",
                "personality": "unknown",
                "appearance": "unknown",
                "quotes": [],
            }
        else:
            current_info = self.context.value[name]

        updates = await self.llm.query_object(
            CharacterInfo,
            CHARACTER=current_info["name"],
            SYNOPSIS=current_info["synopsis"],
            PERSONALITY=current_info["personality"],
            APPEARANCE=current_info["appearance"],
            SETTING_INFO=database_context,
            RECENT_PARAGRAPHS=buffer_context,
            PARAGRAPH=paragraph,
            STORY_STATE=state_context,
            TASK=(
                "You are maintaining the character card for a CHARACTER in a story. "
                "The SETTING_INFO and RECENT_PARAGRAPHS give context for understanding the PARAGRAPH. "
                "Based on the PARAGRAPH, if applicable, suggest updates to the CHARACTER's SYNOPSIS, PERSONALITY, and APPEARANCE. "
                "If applicable suggest new quotes that are indicative of the character's persona."
            ),
        )

        new_info[name] = {
            "name": name,
            "synopsis": updates.updated_synopsis or current_info["synopsis"],
            "personality": updates.updated_personality or current_info["personality"],
            "appearance": updates.updated_appearance or current_info["appearance"],
            "quotes": current_info["quotes"] + (updates.new_quotes or []),
        }

        return new_info

    @horsefunction
    async def update(
        self,
        database_context: Value,
        buffer_context: Value,
        state_context: Value,
        paragraph: Value,
    ) -> AsyncGenerator[Value[dict[str, CharacterInfo]], GradContext]:
        assert isinstance(state_context.value, LiveState)
        updates = await asyncio.gather(
            *[
                self._update_character(
                    character,
                    database_context,
                    buffer_context,
                    state_context,
                    paragraph,
                )
                for character in state_context.value.characters_in_scene
            ]
        )

        current_info = self.context.value.copy()
        for update in updates:
            current_info.update(update)

        self.context = Value(
            current_info,
            predecessors=[database_context, buffer_context, state_context, paragraph],
        )

        grad_context = yield self.context

    @horsefunction
    async def load(
        self, query: Value[str | list[str]], **kwargs
    ) -> AsyncGenerator[Value[dict[str, CharacterInfo]], GradContext]:
        if isinstance(query.value, str):
            q = [query.value]
        else:
            q = query.value

        result = {}
        for name in q:
            if name not in self.context.value:
                continue
            result[name] = self.context.value[name]

        grad_context = yield Value(result, predecessors=[query, self.context.value])


@pytest.mark.asyncio
async def test_generate_character_card(reasoning_llm: AsyncLLMEngine):
    story_paragraphs = STORY.split("\n")

    reader = StoryReader(reasoning_llm)
    characters = CharacterCardContext(reasoning_llm)

    # When reading, have it maintain details character info
    # Probably need to add an llm query to do this

    for p in story_paragraphs:
        # Figure out what's new in the paragraph
        p = Value(p)
        loss: ReadResult = await reader.read(p)
        await characters.update(
            loss.database_context, loss.buffer_context, loss.state_context, p
        )
        print(characters.context.value)

        # Update the reader's state based on what was read
        gradients = await loss.backward(reader.parameters())
        await step(gradients)

    assert "James" in characters.context.value
    assert "David" in characters.context.value
