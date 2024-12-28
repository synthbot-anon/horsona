import os
import warnings

import pytest
from dotenv import load_dotenv

from horsona.config import load_indices, load_llms

load_dotenv()


class FixtureFunctionWrapper:
    def __init__(self, name, obj):
        self.__name__ = name
        self.obj = obj

    def __call__(self):
        return self.obj


if os.path.exists("llm_config.json"):
    llm_engines = load_llms()
    for key, engine in llm_engines.items():
        globals()[key] = pytest.fixture(scope="session", autouse=False)(
            FixtureFunctionWrapper(key, engine)
        )
else:
    warnings.warn("LLM config file not found. Skipping LLM fixtures.")

if os.path.exists("index_config.json"):
    indices = load_indices()
    for key, index in indices.items():
        globals()[key] = pytest.fixture(scope="session", autouse=False)(
            FixtureFunctionWrapper(key, index)
        )
else:
    warnings.warn("Index config file not found. Skipping index fixtures.")
