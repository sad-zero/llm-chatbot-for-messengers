from typing import Generator

import pytest
from fastapi.testclient import TestClient
from llm_chatbot_for_messengers.messenger.kakao.api import app
from llm_chatbot_for_messengers.messenger.kakao.container import DaoContainer
from starlette.routing import _DefaultLifespan  # noqa: PLC2701


@pytest.fixture(scope='session')
def client() -> Generator[TestClient, None, None]:
    app.router.lifespan_context = _DefaultLifespan(app.router)  # Ignore app's lifespan
    app.user_middleware = []
    dao_container = DaoContainer()
    dao_container.check_dependencies()
    dao_container.wire(
        modules=[
            'llm_chatbot_for_messengers.messenger.kakao.container',
        ]
    )
    with TestClient(app) as client:
        yield client
