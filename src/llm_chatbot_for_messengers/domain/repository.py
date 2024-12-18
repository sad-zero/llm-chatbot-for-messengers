from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeAlias

from langgraph.checkpoint.base import BaseCheckpointSaver

if TYPE_CHECKING:
    from llm_chatbot_for_messengers.domain.chatbot import Chatbot
    from llm_chatbot_for_messengers.domain.messenger import Messenger
    from llm_chatbot_for_messengers.domain.specification import ChatbotSpecification, MessengerSpecification
    from llm_chatbot_for_messengers.domain.tracing import Tracing


class ChatbotRepository(ABC):
    @abstractmethod
    async def find_chatbot(self, spec: ChatbotSpecification) -> Chatbot:
        pass


class MessengerRepository(ABC):
    @abstractmethod
    async def find_messenger(self, spec: MessengerSpecification) -> Messenger:
        pass


class TracingRepository(ABC):
    @abstractmethod
    async def add_tracing(self, tracing: Tracing) -> None:
        pass


MemoryType: TypeAlias = BaseCheckpointSaver


class MemoryManager(ABC):
    @abstractmethod
    async def acquire_memory(self) -> MemoryType:
        pass

    @abstractmethod
    async def release_memory(self) -> None:
        pass
