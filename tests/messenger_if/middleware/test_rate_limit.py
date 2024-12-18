from asyncio import sleep

import pytest
from llm_chatbot_for_messengers.domain.messenger import Messenger, User
from llm_chatbot_for_messengers.domain.vo import MessengerId, UserId
from llm_chatbot_for_messengers.messenger_if.middleware.rate_limit import InMemoryTokenBucketRateLimitStrategy
from llm_chatbot_for_messengers.messenger_if.vo import MessengerRequest


@pytest.mark.asyncio
async def test_token_bucket_rate_limit():
    # given
    limit: int = 3
    period: int = 3
    strategy = InMemoryTokenBucketRateLimitStrategy(limit=limit, period=period)

    user = User(user_id=UserId(user_id='test_id'))
    messenger: Messenger = Messenger(
        messenger_id=MessengerId(messenger_id='test-messenger'), messenger_name='test-messenger'
    )
    request = MessengerRequest(user=user, messenger=messenger)
    # when & then
    for _ in range(limit):
        assert (await strategy.accept(request))[0] is True
    assert (await strategy.accept(request))[0] is False
    await sleep(period // limit)
    assert (await strategy.accept(request))[0] is True
