from __future__ import annotations

from llm_chatbot_for_messengers.domain.messenger import Messenger, User
from llm_chatbot_for_messengers.domain.vo import MessengerId, UserId


def test_messenger_register():
    # given
    users: list[User] = [User(user_id=UserId(user_id=f'test_id_{idx}')) for idx in range(10)]
    messenger: Messenger = Messenger(
        messenger_id=MessengerId(messenger_id='test-messenger'), messenger_name='test-messenger'
    )
    # when
    registered_users: tuple[User, ...] = messenger.register_users(*users)
    # then
    assert len(users) == len(registered_users)
    for user, registered in zip(users, registered_users):
        assert user != registered
        assert registered.messenger_id == messenger.messenger_id


def test_messenger_get():
    # given
    users: list[User] = [User(user_id=UserId(user_id=f'test_id_{idx}')) for idx in range(10)]
    messenger: Messenger = Messenger(
        messenger_id=MessengerId(messenger_id='test-messenger'), messenger_name='test-messenger'
    )
    messenger.register_users(*users)
    expected = 10
    # when
    actual: int = len(messenger.get_users())
    # then
    assert expected == actual
