from typing import Generator

import pytest
from fastapi.testclient import TestClient
from llm_chatbot_for_messengers.messenger.kakao.api import app


@pytest.fixture(scope='session')
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as client:
        yield client
