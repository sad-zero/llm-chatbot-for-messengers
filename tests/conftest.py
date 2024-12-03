import pytest
from llm_chatbot_for_messengers.core.entity.agent import QAAgent
from llm_chatbot_for_messengers.core.entity.user import User
from llm_chatbot_for_messengers.core.error import SpecificationError
from typing_extensions import override


@pytest.fixture(scope='session')
def fake_agent() -> QAAgent:
    """QAAgent with FakeLLM"""

    class FakeQAAgent(QAAgent):
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


@pytest.fixture(scope='session')
def fake_agent_fallback() -> QAAgent:
    """QAAgent with FakeLLM"""

    class FakeQAAgent(QAAgent):
        @override
        async def _ask(self, user: User, question: str) -> str:
            raise SpecificationError

        @override
        async def _fallback(self, user: User, question: str) -> str:
            return 'Fallback message'

    return FakeQAAgent()
