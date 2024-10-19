import pytest
from horsona.autodiff.variables import Value
from horsona.cache.db_cache import DatabaseCache
from horsona.cache.list_cache import ListCache
from horsona.database.embedding_database import EmbeddingDatabase
from horsona.index.hnsw_index import HnswEmbeddingIndex
from horsona.io.reader import ReadContext, ReaderModule
from horsona.stories.character_card import CharacterCardContext
from pydantic import BaseModel

STORY = """James looked skeptically at his friend David as he sat down at computer #12.
David had won the Hasbro raffle for one of fifteen all-expenses-paid trips for two to Pawtucket, Rhode Island to play the first alpha build of the official My Little Pony MMO: Equestria Online. Hasbro had claimed that a game that revolved so heavily around friendship needed actual friends to test properly.
“Look, I’m glad you invited me,” James said as he picked up his head set. “And it’s cool that I get to try something before anyone else. But I’m still not sure I really want to play a game that’s so...so...pink and purple.”
David scoffed. “I know you. What was that Korean MMO with the little girls that you were so into?  You’ll be running dungeons over and over again, just like you always do with every MMO that comes out. But me?” David gave the widest smile possible. “I’m here for the ponies. We both get what we want!”"""


class LiveState(BaseModel):
    current_location: str = "unknown"
    characters_in_scene: list[str] = []
    last_speaker: str = "none"
    expected_next_speaker: str = "none"


@pytest.mark.asyncio
async def test_reader(reasoning_llm, query_index: HnswEmbeddingIndex):
    story_paragraphs = STORY.split("\n")
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

    for p in story_paragraphs:
        # Figure out what's new in the paragraph
        read_context, read_context_loss = await reader.read(
            read_context, Value("Story paragraph", p)
        )

        # Update the reader's state based on what was read
        await read_context_loss.step([setting_db])

    context_state = read_context.state_dict()
    read_context = ReadContext.load_state_dict(
        context_state,
        args={
            "database_context": {
                "llm": reasoning_llm,
                "database": setting_db,
            }
        },
    )

    assert len(read_context.buffer_context) > 0
    assert len(read_context.database_context) > 0
    assert read_context.state_context.value.last_speaker == "David"
    assert set(read_context.state_context.value.characters_in_scene) == {
        "James",
        "David",
    }

    assert (
        "computer #12" in read_context.state_context.value.current_location.lower()
        or "pawtucket" in read_context.state_context.value.current_location.lower()
    )


@pytest.mark.asyncio
async def test_generate_character_card(reasoning_llm, query_index):
    story_paragraphs = STORY.split("\n")
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

    characters = CharacterCardContext(reasoning_llm)

    # When reading, have it maintain details character info
    # Probably need to add an llm query to do this

    for p in story_paragraphs:
        # Figure out what's new in the paragraph
        read_context, read_context_loss = await reader.read(
            read_context, Value("Story paragraph", p)
        )
        await characters.update(
            read_context.state_context.value.characters_in_scene, read_context, p
        )

        # Update the reader's state based on what was read
        await read_context_loss.step([setting_db])

    assert "James" in characters
    assert "David" in characters
