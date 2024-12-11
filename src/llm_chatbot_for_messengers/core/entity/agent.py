"""
Domain
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, Field, PrivateAttr
from typing_extensions import override

from llm_chatbot_for_messengers.core.entity.user import User
from llm_chatbot_for_messengers.core.error import SpecificationError
from llm_chatbot_for_messengers.core.specification import (
    check_timeout,
)
from llm_chatbot_for_messengers.core.vo import (
    AgentConfig,
    QAState,
)
from llm_chatbot_for_messengers.core.workflow.qa import QAWorkflow

if TYPE_CHECKING:
    from llm_chatbot_for_messengers.core.output.memory import MemoryType


logger = logging.getLogger(__name__)


class QAAgent(ABC):
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the agent.
        - Acquire memory
        - etc
        """

    @abstractmethod
    async def shutdown(self) -> None:
        """
        Shutdown the agent.
        - Release memory
        - etc
        """

    async def ask(self, user: User, question: str, timeout: int | None = None) -> str:
        """Ask question

        Args:
            user          (User): User Information
            question       (str): User's question
            timeout (int | None): Timeout seconds
        Returns:
            str: Agent's answer
        """
        if not isinstance(user, User) or not isinstance(question, str):
            err_msg: str = f'Invalid arguments: user: {user}, question: {question}'
            raise TypeError(err_msg)
        try:
            if timeout is None:
                return await self._ask(user=user, question=question)
            traced = check_timeout(self.__class__._ask, timeout=timeout)  # noqa: SLF001
            return await traced(self=self, user=user, question=question)
        except SpecificationError:
            logger.exception('')
            return await self._fallback(user=user, question=question)

    @abstractmethod
    async def _ask(self, user: User, question: str) -> str:
        """Answer user's question
        Args:
            user    (User): User Information
            question (str): User's question
        Returns:
            str: Agent's answer
        """

    @abstractmethod
    async def _fallback(self, user: User, question: str) -> str:
        """Fallback message based on user's question(**Return immidiately**)
        Args:
            user    (User): User Information
            question (str): User's question
        Returns:
            str: Static fallback message
        """


class QAAgentImpl(BaseModel, QAAgent):
    config: AgentConfig = Field(description='Agent configuration')
    __workflow: QAWorkflow = PrivateAttr(default=None)  # type: ignore

    @override
    async def initialize(self) -> None:
        if self.config.global_configs.memory_manager is not None:
            memory: MemoryType | None = await self.config.global_configs.memory_manager.acquire_memory()
        else:
            memory = None
        self.__workflow = QAWorkflow.get_instance(config=self.config.node_configs, memory=memory)

    @override
    async def shutdown(self) -> None:
        if self.config.global_configs.memory_manager is not None:
            await self.config.global_configs.memory_manager.release_memory()

    @override
    async def _ask(self, user: User, question: str) -> str:
        initial: QAState = QAState.put_question(question=question)
        response: QAState = await self.__workflow.ainvoke(
            initial,
            config={
                'run_name': 'QAAgent.ask',
                'metadata': {
                    'user': user.model_dump(
                        mode='python',
                    )
                },
                'configurable': {'thread_id': user.user_id.user_id},
            },
        )  # type: ignore
        result: str = cast(str, response.answer)
        return result

    @override
    async def _fallback(self, user: User, question: str) -> str:
        log_info = {
            'user': user.model_dump(),
            'question': question,
        }
        log_msg: str = f'fallback: {log_info}'
        logger.warning(log_msg)
        return self.config.global_configs.fallback_message
