from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator


class Messenger(BaseModel):
    messenger_id: MessengerId = Field(description="Messenger's Id")
    messenger_name: str = Field(description="Messenger's name")
    __users: list[User] = PrivateAttr(default_factory=list)

    def get_users(self) -> tuple[User, ...]:
        """Get messenger's users
        Returns:
            list[User]: View
        """
        return tuple(self.__users)

    def register_users(self, user: User, *users: User) -> tuple[User, ...]:
        """Register user to messenger
        Args:
            user (User): Not registered user
        Returns:
            tuple[User]: Registered users
        """
        users += (user,)
        registered: list[User] = []
        for _user in users:
            if not isinstance(_user, User):
                err_msg: str = f'user({_user}) should be User type.'
                raise TypeError(err_msg)
            registered_user: User = User(
                messenger_id=self.messenger_id, user_id=_user.user_id, user_name=_user.user_name
            )
            registered.append(registered_user)
        self.__users.extend(registered)
        return tuple(registered)


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


class User(BaseModel):
    messenger_id: MessengerId | None = Field(description="Messenger's Id", default=None)
    user_id: UserId = Field(description="User's Unique Id")
    user_name: str | None = Field(description="User's name", default=None)


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
