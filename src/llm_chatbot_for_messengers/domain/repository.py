from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_chatbot_for_messengers.domain.chatbot import Chatbot
    from llm_chatbot_for_messengers.domain.specification import ChatbotSpecification


class ChatbotRepository(ABC):
    @abstractmethod
    async def find_chatbot(self, spec: ChatbotSpecification) -> Chatbot:
        pass
