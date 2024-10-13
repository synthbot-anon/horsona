import json

import pytest
from dotenv import load_dotenv
from horsona.index import indices_from_config
from horsona.llm import engines_from_config

load_dotenv()


# Add LLMs from llm_config.json
with open("llm_config.json") as f:
    config = json.load(f)
    engines = engines_from_config(config)

for key, engine in engines.items():
    globals()[key] = pytest.fixture(scope="session", autouse=False)(lambda: engine)


# Add indices from index_config.json
with open("index_config.json") as f:
    config = json.load(f)
    indices = indices_from_config(config)

for key, index in indices.items():
    globals()[key] = pytest.fixture(scope="session", autouse=False)(lambda: index)
