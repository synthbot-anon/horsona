import os

import pytest
from dotenv import load_dotenv
from horsona.index import indices, load_indices
from horsona.llm import engines as llm_engines
from horsona.llm import load_engines as load_llm_engines

load_dotenv()


def get_missing_config_files():
    missing_files = []
    if not os.path.exists("llm_config.json"):
        missing_files.append("llm_config.json")
    if not os.path.exists("index_config.json"):
        missing_files.append("index_config.json")
    return missing_files


missing_files = get_missing_config_files()
if missing_files:
    error_msg = (
        "Required config files not found in current working directory. "
        "Run dev-install.sh to create them, or check the README for how to create them. "
        f"Missing: {' and '.join(missing_files)}"
    )
    raise FileNotFoundError(error_msg)


class FixtureFunctionWrapper:
    def __init__(self, name, obj):
        self.__name__ = name
        self.obj = obj

    def __call__(self):
        return self.obj


load_llm_engines()
for key, engine in llm_engines.items():
    globals()[key] = pytest.fixture(scope="session", autouse=False)(
        FixtureFunctionWrapper(key, engine)
    )


load_indices()
for key, index in indices.items():
    globals()[key] = pytest.fixture(scope="session", autouse=False)(
        FixtureFunctionWrapper(key, index)
    )
