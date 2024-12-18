import pytest
from llm_chatbot_for_messengers.domain.output.dao import InMemoryMessengerDaoImpl
from llm_chatbot_for_messengers.domain.vo import MessengerIdEnum


@pytest.mark.asyncio
async def test_inmemory_messenger_dao():
    # given
    dao = InMemoryMessengerDaoImpl()
    messenger_id = MessengerIdEnum.KAKAO.value
    # when
    messenger = await dao.get_messenger(messenger_id=messenger_id)
    # then
    assert messenger.messenger_id == messenger_id
