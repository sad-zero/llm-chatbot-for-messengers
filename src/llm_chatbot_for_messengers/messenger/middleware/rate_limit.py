from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from functools import cached_property
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, PrivateAttr
from typing_extensions import override

from llm_chatbot_for_messengers.core.entity.user import User  # noqa: TCH001

if TYPE_CHECKING:
    from llm_chatbot_for_messengers.core.vo import UserId
    from llm_chatbot_for_messengers.messenger.vo import MessengerRequest


class RateLimitStrategy(ABC):
    @abstractmethod
    async def accept(self, request: MessengerRequest) -> bool:
        """Determine whether this request is valid or not.
        Args:
            request (MessengerRequest): Request information
        Returns:
            bool: True if the request is acceptable else False.
        """

    @abstractmethod
    async def refill(self, user: User, timestamp: datetime) -> int:
        """Refill request limits based on timestamp
        Args:
            user          (User): User information
            timestamp (datetime): Refill timestamp
        Returns:
            int: Refilled limit
        """


class TokenBucketRateLimitStrategy(RateLimitStrategy, BaseModel):
    """Implemented by TokenBucket Algorithm
    # TODO: User by messengers
    """

    limit: int = Field(description='Maximum request number in period', gt=0)
    period: int = Field(description='Seconds to determine the limit is over or not.', gt=1)
    __bucket: dict[UserId, tuple[datetime, int]] = PrivateAttr(default_factory=dict)

    @override
    async def accept(self, request: MessengerRequest) -> bool:
        user: User = request.user
        timestamp = datetime.now(timezone.utc)

        if (tokens := await self.refill(user=user, timestamp=timestamp)) <= 0:
            return False
        self.__bucket[user.user_id] = (timestamp, tokens - 1)
        return True

    @override
    async def refill(self, user: User, timestamp: datetime) -> int:
        if (bucket_info := self.__bucket.get(user.user_id)) is None:
            self.__bucket[user.user_id] = (timestamp, self.limit)
            return self.limit

        last_updated_dt, tokens = bucket_info
        if timestamp < last_updated_dt:
            err_msg: str = f'timestamp({timestamp}) is pass last updated time({last_updated_dt})'
            raise ValueError(err_msg)
        elapsed_sec = (timestamp - last_updated_dt).seconds
        new_tokens: int = elapsed_sec // self.__rate
        updated_token = min(self.limit, tokens + new_tokens)
        self.__bucket[user.user_id] = (timestamp, updated_token)
        return updated_token

    @cached_property
    def __rate(self) -> int:
        """Interval to refill a new token.
        Returns:
            int: lower bound of period/limit
        """
        return self.period // self.limit
