from __future__ import annotations

from enum import Enum, unique
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class UserId(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_seq: int | None = Field(description="User's sequence", default=None)
    user_id: str | None = Field(description="User's unique string", default=None)

    @model_validator(mode='after')
    def check_id(self) -> Self:
        if (self.user_seq, self.user_id) == (None, None):
            err_msg: str = 'One of id or seq should exist.'
            raise ValueError(err_msg)
        return self


class MessengerId(BaseModel):
    model_config = ConfigDict(frozen=True)

    messenger_seq: int | None = Field(description="Messenger's sequence", default=None)
    messenger_id: str | None = Field(description="Messenger's unique string", default=None)

    @model_validator(mode='after')
    def check_id(self) -> Self:
        if (self.messenger_seq, self.messenger_id) == (None, None):
            err_msg: str = 'One of id or seq should exist.'
            raise ValueError(err_msg)
        return self


@unique
class MessengerIdEnum(Enum):
    KAKAO = MessengerId(messenger_seq=1, messenger_id='kakao')
