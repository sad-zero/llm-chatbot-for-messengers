from typing import Generator

import pytest
from fastapi.testclient import TestClient
from llm_chatbot_for_messengers.messenger.kakao.api import app
from llm_chatbot_for_messengers.messenger.kakao.container import get_qa_agent


@pytest.fixture(scope='session')
def client(fake_agent) -> Generator[TestClient, None, None]:
    def _get_fake_qa_agent():
        return fake_agent

    app.dependency_overrides[get_qa_agent] = _get_fake_qa_agent
    with TestClient(app) as client:
        yield client
