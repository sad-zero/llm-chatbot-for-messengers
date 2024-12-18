from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TypeAlias

from langgraph.checkpoint.base import BaseCheckpointSaver

from llm_chatbot_for_messengers.domain.error import SpecificationError
from llm_chatbot_for_messengers.domain.messenger import User
from llm_chatbot_for_messengers.domain.specification import check_timeout

logger = logging.getLogger(__name__)


class Chatbot(ABC):
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


MemoryType: TypeAlias = BaseCheckpointSaver


class MemoryManager(ABC):
    @abstractmethod
    async def acquire_memory(self) -> MemoryType:
        pass

    @abstractmethod
    async def release_memory(self) -> None:
        pass
