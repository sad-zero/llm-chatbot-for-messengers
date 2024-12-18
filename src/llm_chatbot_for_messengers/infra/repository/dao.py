from __future__ import annotations

from abc import ABC, abstractmethod

from typing_extensions import override

from llm_chatbot_for_messengers.domain.error import DataError
from llm_chatbot_for_messengers.domain.messenger import Messenger
from llm_chatbot_for_messengers.domain.vo import MessengerId, MessengerIdEnum


class MessengerDao(ABC):
    @abstractmethod
    async def get_messenger(self, messenger_id: MessengerId) -> Messenger:
        """Find messenger by messenger_id
        Args:
            messenger_id    (str): Messenger's id
        Returns:
            messenger (Messenger): Messenger
        """


class InMemoryMessengerDaoImpl(MessengerDao):
    def __init__(self):
        self.__messengers = {
            MessengerIdEnum.KAKAO.value: Messenger(
                messenger_id=MessengerIdEnum.KAKAO.value, messenger_name='KAKAOTALK'
            ),
        }

    @override
    async def get_messenger(self, messenger_id: MessengerId) -> Messenger:
        if not isinstance(messenger_id, MessengerId):
            err_msg: str = f'messenger_id({messenger_id}) should be MessengerId type'
            raise TypeError(err_msg)
        if messenger_id not in self.__messengers:
            err_msg = f'messenger_id({messenger_id}) not found'
            raise DataError(err_msg)
        return self.__messengers[messenger_id]
