import pytest
from horsona.autodiff.basic import HorseModule
from horsona.autodiff.variables import Value
from horsona.memory.list_cache import ListCache
from pydantic import BaseModel


class PonyName(BaseModel):
    name: str


@pytest.mark.asyncio
async def test_variable(reasoning_llm):
    input_text = Value("Text", "Test Value", reasoning_llm)
    saved = input_text.state_dict()
    restored_text = Value.load_state_dict(saved)
    assert restored_text.value == "Test Value"

    input_text = Value("Name", PonyName(name="Celestia"), reasoning_llm)
    saved = input_text.state_dict()
    restored_text = Value.load_state_dict(saved)
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
    restored_module = SampleModule.load_state_dict(saved)
    assert restored_module.input_text.value == "Test Value"


@pytest.mark.asyncio
async def test_list_cache():
    cache = ListCache(3, [Value("Text", "Test Value")])
    saved = cache.state_dict()

    restored_cache = ListCache.load_state_dict(saved)
    assert restored_cache[0].value == "Test Value"
    assert restored_cache.max_size == 3
