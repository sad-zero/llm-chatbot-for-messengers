from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from functools import cached_property
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, PrivateAttr
from typing_extensions import override

from llm_chatbot_for_messengers.domain.vo import MessengerId  # noqa: TCH001

if TYPE_CHECKING:
    from llm_chatbot_for_messengers.domain.messenger import Messenger, User
    from llm_chatbot_for_messengers.domain.vo import UserId
    from llm_chatbot_for_messengers.messenger_if.vo import MessengerRequest

logger = logging.getLogger(__name__)


class RateLimitStrategy(ABC):
    @abstractmethod
    async def accept(self, request: MessengerRequest) -> tuple[bool, int]:
        """Determine whether this request is valid or not.
        Args:
            request (MessengerRequest): Request information
        Returns:
            tuple[bool, int]: True if the request is acceptable else False. Next available seconds.
        """

    @abstractmethod
    async def refill(self, messenger: Messenger, user: User, timestamp: datetime) -> int:
        """Refill request limits based on timestamp
        Args:
            messenger(Messenger): Messenger information
            user          (User): User information
            timestamp (datetime): Refill timestamp
        Returns:
            int: Refilled limit
        """


class InMemoryTokenBucketRateLimitStrategy(RateLimitStrategy, BaseModel):
    """Implemented by TokenBucket Algorithm"""

    limit: int = Field(description='Maximum request number in period', gt=0)
    period: int = Field(description='Seconds to determine the limit is over or not.', gt=1)
    __bucket: dict[tuple[MessengerId, UserId], tuple[datetime, int]] = PrivateAttr(default_factory=dict)

    @override
    async def accept(self, request: MessengerRequest) -> tuple[bool, int]:
        messenger: Messenger = request.messenger
        user: User = request.user
        timestamp = datetime.now(timezone.utc)
        key = self.__get_key(messenger=messenger, user=user)

        if (tokens := await self.refill(messenger=messenger, user=user, timestamp=timestamp)) <= 0:
            last_updated_dt, _ = self.__bucket[key]
            remains = self.__rate - (timestamp - last_updated_dt).seconds
            return False, remains
        self.__bucket[key] = (timestamp, tokens - 1)

        log_msg: str = f'{user}: {tokens - 1}'
        logger.info(log_msg)
        return True, 0

    @override
    async def refill(self, messenger: Messenger, user: User, timestamp: datetime) -> int:
        key = self.__get_key(messenger=messenger, user=user)
        if (bucket_info := self.__bucket.get(key)) is None:
            self.__bucket[key] = (timestamp, self.limit)
            return self.limit

        last_updated_dt, tokens = bucket_info
        if timestamp < last_updated_dt:
            err_msg: str = f'timestamp({timestamp}) is pass last updated time({last_updated_dt})'
            raise ValueError(err_msg)
        elapsed_sec = (timestamp - last_updated_dt).seconds
        new_tokens: int = elapsed_sec // self.__rate
        if new_tokens == 0:
            return tokens

        updated_token = min(self.limit, tokens + new_tokens)
        self.__bucket[key] = (timestamp, updated_token)
        return updated_token

    @cached_property
    def __rate(self) -> int:
        """Interval to refill a new token.
        Returns:
            int: lower bound of period/limit
        """
        return self.period // self.limit

    @classmethod
    def __get_key(cls, messenger: Messenger, user: User) -> tuple[MessengerId, UserId]:
        return messenger.messenger_id, user.user_id
