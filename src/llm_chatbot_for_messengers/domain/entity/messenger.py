from __future__ import annotations

from pydantic import BaseModel, Field, PrivateAttr

from llm_chatbot_for_messengers.domain.entity.user import User
from llm_chatbot_for_messengers.domain.vo import MessengerId  # noqa: TCH001


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
