import pytest
from llm_chatbot_for_messengers.domain.chatbot import Chatbot
from llm_chatbot_for_messengers.domain.error import SpecificationError
from llm_chatbot_for_messengers.domain.messenger import User
from typing_extensions import override


@pytest.fixture
def fake_agent() -> Chatbot:
    """QAAgent with FakeLLM"""

    class FakeQAAgent(Chatbot):
        @override
        async def initialize(self):
            pass

        @override
        async def shutdown(self):
            pass

        @override
        async def _ask(self, user: User, question: str) -> str:
            answer: str = f"""
            user: {user}
            question: {question}
            """.strip()
            return answer

        @override
        async def _fallback(self, user: User, question: str) -> str:
            return 'Fallback message'

    return FakeQAAgent()


@pytest.fixture
def fake_agent_fallback() -> Chatbot:
    """QAAgent with FakeLLM"""

    class FakeQAAgent(Chatbot):
        @override
        async def initialize(self):
            pass

        @override
        async def shutdown(self):
            pass

        @override
        async def _ask(self, user: User, question: str) -> str:
            raise SpecificationError

        @override
        async def _fallback(self, user: User, question: str) -> str:
            return 'Fallback message'

    return FakeQAAgent()
