from typing import Generator

import pytest
from fastapi.testclient import TestClient
from llm_chatbot_for_messengers.messenger.kakao.api import app
from llm_chatbot_for_messengers.messenger.kakao.container import get_qa_agent


@pytest.fixture(scope='session')
def client(fake_agent) -> Generator[TestClient, None, None]:
    app.dependency_overrides[get_qa_agent] = lambda: fake_agent
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope='session')
def failed_client(fake_agent_fallback) -> Generator[TestClient, None, None]:
    app.dependency_overrides[get_qa_agent] = lambda: fake_agent_fallback
    with TestClient(app) as client:
        yield client
