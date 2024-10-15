import pytest
from horsona.autodiff.basic import HorseModule, load_state_dict, step
from horsona.autodiff.functions import extract_object
from horsona.autodiff.losses import apply_loss
from horsona.autodiff.variables import Value
from horsona.cache.db_cache import DatabaseCache
from horsona.cache.list_cache import ListCache
from horsona.cache.value_cache import ValueCache
from horsona.database.embedding_database import EmbeddingDatabase
from pydantic import BaseModel


class PonyName(BaseModel):
    name: str


@pytest.mark.asyncio
async def test_variable(reasoning_llm):
    input_text = Value("Text", "Test Value", reasoning_llm)
    saved = input_text.state_dict()
    print(saved)
    restored_text = Value.load_state_dict(saved, args={"updater_llm": reasoning_llm})
    print(restored_text.datatype, restored_text.value)
    assert restored_text.value == "Test Value"

    input_text = Value("Name", PonyName(name="Celestia"), reasoning_llm)
    saved = input_text.state_dict()
    restored_text = Value.load_state_dict(saved, args={"updater_llm": reasoning_llm})
    assert restored_text.value.name == "Celestia"


class SampleModule(HorseModule):
    def __init__(
        self, value: str = None, reasoning_llm=None, input_text=None, **kwargs
    ):
        super().__init__(**kwargs)
        if input_text:
            self.input_text = input_text
        else:
            self.input_text = Value("Text", value, reasoning_llm)


@pytest.mark.asyncio
async def test_module(reasoning_llm):
    module = SampleModule("Test Value")
    saved = module.state_dict()
    restored_module = SampleModule.load_state_dict(
        saved,
        args={
            "reasoning_llm": reasoning_llm,
        },
    )
    assert restored_module.input_text.value == "Test Value"


@pytest.mark.asyncio
async def test_value_cache():
    cache = ValueCache(Value("Text", "Test Value"))
    saved = cache.state_dict()

    restored_cache = ValueCache.load_state_dict(saved)
    assert restored_cache.context.value == "Test Value"


@pytest.mark.asyncio
async def test_list_cache():
    cache = ListCache(3, [Value("Text", "Test Value")])
    saved = cache.state_dict()

    restored_cache = ListCache.load_state_dict(saved)
    assert restored_cache.context[0].value == "Test Value"
    assert restored_cache.size == 3


@pytest.mark.asyncio
async def test_db_cache(reasoning_llm, query_index):
    database = EmbeddingDatabase(
        reasoning_llm,
        query_index,
    )

    await database.insert(
        {
            "Test Key1": "Test Value1",
            "Test Key2": "Test Value2",
        }
    )

    cache = DatabaseCache(reasoning_llm, database, 3)

    saved = cache.state_dict()
    cache = DatabaseCache.load_state_dict(
        saved,
        args={
            "llm": reasoning_llm,
            "database": database,
        },
    )
    context = await cache.load(Value("Key", "Test Key1"))
    assert "Test Key1" in context.data
    assert "Test Key2" not in context.data

    saved = cache.state_dict()
    cache = DatabaseCache.load_state_dict(
        saved,
        args={
            "llm": reasoning_llm,
            "database": {
                "llm": reasoning_llm,
                "index": query_index,
            },
        },
    )
    context = await cache.load(Value("Key", "Test Key2"))
    assert "Test Key1" in context.data
    assert "Test Key2" in context.data
