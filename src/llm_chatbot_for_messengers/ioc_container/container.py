from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject

from llm_chatbot_for_messengers.domain.chatbot import Chatbot, ChatbotImpl
from llm_chatbot_for_messengers.domain.configuration import AgentConfig, LLMConfig
from llm_chatbot_for_messengers.domain.output.dao import InMemoryMessengerDaoImpl, MessengerDao
from llm_chatbot_for_messengers.domain.output.memory import VolatileMemoryManager
from llm_chatbot_for_messengers.messenger_if.middleware.rate_limit import (
    InMemoryTokenBucketRateLimitStrategy,
    RateLimitStrategy,
)


class AgentContainer(containers.DeclarativeContainer):
    qa_agent: providers.Singleton[Chatbot] = providers.ThreadSafeSingleton(
        ChatbotImpl,
        config=AgentConfig.builder()
        .add_node(
            node_name='answer_node',
            template_name='kakao_v3',
            llm_config=LLMConfig(model='gpt-4o-2024-11-20', temperature=0.52, max_tokens=200),
        )
        .add_node(
            node_name='summary_node',
            template_name='kakao_v2',
            llm_config=LLMConfig(model='gpt-4o-mini-2024-07-18', temperature=0.4, max_tokens=200),
        )
        .add_fallback('미안해용. ㅠㅠ 질문이 너무 어려워용..')
        .add_memory_manager(memory_manager=VolatileMemoryManager())
        #   .add_memory_manager(memory_manager=PersistentMemoryManager(conn_uri=os.getenv('CORE_DB_URI'))),  # type: ignore
        .build(),
    )


class DaoContainer(containers.DeclarativeContainer):
    messenger_dao: providers.Singleton[MessengerDao] = providers.ThreadSafeSingleton(
        InMemoryMessengerDaoImpl,
    )


class MiddlewareContainer(containers.DeclarativeContainer):
    rate_limit: providers.Singleton[RateLimitStrategy] = providers.ThreadSafeSingleton(
        InMemoryTokenBucketRateLimitStrategy,
        limit=100,
        period=86400,  # 1 day
    )


_qa_agent: Chatbot = Provide[AgentContainer.qa_agent]


def get_qa_agent() -> Chatbot:
    return _qa_agent


@inject
def get_rate_limit_strategy(
    rate_limit_strategy: RateLimitStrategy = Provide[MiddlewareContainer.rate_limit],
) -> RateLimitStrategy:
    return rate_limit_strategy


@inject
def _get_messenger_dao(messenger_dao: MessengerDao = Provide[DaoContainer.messenger_dao]) -> MessengerDao:
    return messenger_dao


def get_messenger_dao() -> MessengerDao:
    return _get_messenger_dao()
