import pytest
from horsona.autodiff.basic import step
from horsona.autodiff.variables import Value
from horsona.database.embedding_database import EmbeddingDatabase
from horsona.index.hnsw_index import HnswEmbeddingIndex
from horsona.stories.character_card import CharacterCardContext
from horsona.stories.reader import ReaderModule

STORY = """James looked skeptically at his friend David as he sat down at computer #12.
David had won the Hasbro raffle for one of fifteen all-expenses-paid trips for two to Pawtucket, Rhode Island to play the first alpha build of the official My Little Pony MMO: Equestria Online. Hasbro had claimed that a game that revolved so heavily around friendship needed actual friends to test properly.
“Look, I’m glad you invited me,” James said as he picked up his head set. “And it’s cool that I get to try something before anyone else. But I’m still not sure I really want to play a game that’s so...so...pink and purple.”
David scoffed. “I know you. What was that Korean MMO with the little girls that you were so into?  You’ll be running dungeons over and over again, just like you always do with every MMO that comes out. But me?” David gave the widest smile possible. “I’m here for the ponies. We both get what we want!”"""


@pytest.mark.asyncio
async def test_reader(reasoning_llm, query_index: HnswEmbeddingIndex):
    story_paragraphs = STORY.split("\n")
    setting_db = EmbeddingDatabase(
        reasoning_llm,
        query_index,
        requires_grad=True,
    )
    reader = ReaderModule(reasoning_llm, setting_db)

    for p in story_paragraphs:
        # Figure out what's new in the paragraph
        read_context, read_context_loss = await reader.read(Value(p))

        # Update the reader's state based on what was read
        gradients = await read_context_loss.backward(reader.parameters())
        await step(gradients)

    reader_state = reader.state_dict()
    reader = ReaderModule.load_state_dict(
        reader_state,
        args={
            "llm": reasoning_llm,
            "database_cache": {
                "llm": reasoning_llm,
                "database": {
                    "llm": reasoning_llm,
                    "index": {
                        "model": query_index.model,
                    },
                },
            },
        },
    )

    assert len(reader.buffer_cache.context) > 0
    assert len(reader.database_cache.context) > 0
    assert reader.live_cache.context.value.last_speaker == "David"
    assert set(reader.live_cache.context.value.characters_in_scene) == {
        "James",
        "David",
    }

    assert (
        "computer #12" in reader.live_cache.context.value.current_location.lower()
        or "pawtucket" in reader.live_cache.context.value.current_location.lower()
    )


@pytest.mark.asyncio
async def test_generate_character_card(reasoning_llm, query_index):
    story_paragraphs = STORY.split("\n")
    setting_db = EmbeddingDatabase(
        reasoning_llm,
        query_index,
        requires_grad=True,
    )
    reader = ReaderModule(reasoning_llm, setting_db)
    characters = CharacterCardContext(reasoning_llm)

    # When reading, have it maintain details character info
    # Probably need to add an llm query to do this

    for p in story_paragraphs:
        # Figure out what's new in the paragraph
        p = Value(p)
        read_context, read_context_loss = await reader.read(p)
        await characters.update(
            read_context.state_context.value.characters_in_scene, read_context, p
        )
        print(characters.context.value)

        # Update the reader's state based on what was read
        gradients = await read_context_loss.backward(reader.parameters())
        await step(gradients)

    assert "James" in characters.context.value
    assert "David" in characters.context.value
