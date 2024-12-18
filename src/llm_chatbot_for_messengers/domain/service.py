from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_chatbot_for_messengers.domain.chatbot import Chatbot
    from llm_chatbot_for_messengers.domain.factory import TracingFactory
    from llm_chatbot_for_messengers.domain.messenger import Messenger
    from llm_chatbot_for_messengers.domain.repository import MessengerRepository, TracingRepository
    from llm_chatbot_for_messengers.domain.specification import ChatSpecification, MessengerSpecification


class ChatService:
    def __init__(self, tracing_factory: TracingFactory, tracing_repository: TracingRepository):
        self.__tracing_factory = tracing_factory
        self.__tracing_repository = tracing_repository

    async def chat(self, chatbot: Chatbot, messenger: Messenger, spec: ChatSpecification) -> str:
        """Chat and trace
        Args:
            chatbot        (Chatbot): Initialized chatbot
            messenger    (Messenger): Initialized messenger
            spec (ChatSpecification): Chat Specification
        """
        # TODO: impl
        raise NotImplementedError


class MessengerService:
    def __init__(self, messenger_repository: MessengerRepository):
        self.__messenger_repository = messenger_repository

    async def find_messenger(self, spec: MessengerSpecification) -> Messenger:
        """Find messenger
        Args:
            spec (MessengerSpecification): Messenger Specification
        Returns:
            Messenger: Initialized messenger
        """
        # TODO: impl
        raise NotImplementedError
