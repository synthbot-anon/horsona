import pytest
from dotenv import load_dotenv


@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    load_dotenv()