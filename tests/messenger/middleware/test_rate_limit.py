from asyncio import sleep

import pytest
from llm_chatbot_for_messengers.core.entity.user import User
from llm_chatbot_for_messengers.core.vo import UserId
from llm_chatbot_for_messengers.messenger.middleware.rate_limit import InMemoryTokenBucketRateLimitStrategy
from llm_chatbot_for_messengers.messenger.vo import MessengerRequest


@pytest.mark.asyncio
async def test_token_bucket_rate_limit():
    # given
    limit: int = 3
    period: int = 3
    strategy = InMemoryTokenBucketRateLimitStrategy(limit=limit, period=period)

    user = User(user_id=UserId(user_id='test_id'))
    messenger: str = 'test-messenger'
    request = MessengerRequest(user=user, messenger=messenger)
    # when & then
    for _ in range(limit):
        assert await strategy.accept(request) is True
    assert await strategy.accept(request) is False
    await sleep(period // limit)
    assert await strategy.accept(request) is True
