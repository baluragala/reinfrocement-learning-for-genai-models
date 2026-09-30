import pytest

import rllab as rl
from fake_llm import FakeBackend


@pytest.fixture(autouse=True)
def fake_backend():
    rl.models.set_backend(FakeBackend())
    yield
    rl.models.set_backend(None)
