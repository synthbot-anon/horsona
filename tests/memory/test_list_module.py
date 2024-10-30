import pytest
from horsona.autodiff.variables import Value
from horsona.llm.engine_utils import compile_user_prompt
from horsona.memory.list_module import ListModule


@pytest.mark.asyncio
async def test_list_module_max_length():
    # Create list module with max length of 20
    single_item_length = len(await compile_user_prompt(ITEM=Value("Text", "xxxxx")))
    max_length = single_item_length * 4
    list_module = ListModule(max_length=max_length)

    # Add items that will exceed max length
    await list_module.append(Value("Text", "12345"))  # Length 5
    await list_module.append(Value("Text", "67890"))  # Length 5
    await list_module.append(Value("Text", "abcde"))  # Length 5
    await list_module.append(Value("Text", "fghij"))  # Length 5
    await list_module.append(Value("Text", "klmno"))  # Length 5

    # Should only keep last 4 items to stay under max_length of 20
    assert len(list_module.items) < 5

    assert list_module.items[0].value != "12345"
    assert list_module.items[-1].value == "klmno"

    # Total length should be <= max_length
    assert sum(list_module.item_lengths) <= max_length


@pytest.mark.asyncio
async def test_list_module_clear():
    list_module = ListModule()
    await list_module.append(Value("Text", "test1"))
    await list_module.append(Value("Text", "test2"))

    assert len(list_module.items) == 2

    list_module.clear()
    assert len(list_module.items) == 0
    assert list_module.item_lengths is None


@pytest.mark.asyncio
async def test_list_module_get_items():
    list_module = ListModule()
    item1 = Value("Text", "test1")
    item2 = Value("Text", "test2")

    await list_module.append(item1)
    await list_module.append(item2)

    items = list_module.get_items()
    assert len(items) == 2
    assert items[0].value == "test1"
    assert items[1].value == "test2"


@pytest.mark.asyncio
async def test_list_module_init_with_items():
    items = [Value("Text", "test1"), Value("Text", "test2")]
    list_module = ListModule(items=items, max_length=100)

    assert len(list_module.items) == 2
    assert list_module.items[0].value == "test1"
    assert list_module.max_length == 100
    assert list_module.item_lengths is None  # Should be computed on first append
