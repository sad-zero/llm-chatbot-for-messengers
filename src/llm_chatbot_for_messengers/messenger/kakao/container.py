import os
from contextlib import asynccontextmanager

from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject
from fastapi import FastAPI

from llm_chatbot_for_messengers.core.entity.agent import QAAgent, QAAgentImpl
from llm_chatbot_for_messengers.core.output.memory import PersistentMemoryManager
from llm_chatbot_for_messengers.core.vo import LLMConfig, WorkflowGlobalConfig, WorkflowNodeConfig


class AgentContainer(containers.DeclarativeContainer):
    qa_agent: providers.Singleton[QAAgent] = providers.ThreadSafeSingleton(
        QAAgentImpl,
        workflow_configs={
            'answer_node': WorkflowNodeConfig(
                node_name='answer_node',
                template_name='kakao_v2',
                llm_config=LLMConfig(model='gpt-4o-2024-08-06', temperature=0.52, max_tokens=200),
            )
        },
        global_configs=WorkflowGlobalConfig(
            fallback_message='미안해용. ㅠㅠ 질문이 너무 어려워용..',
            memory_manager=PersistentMemoryManager(conn_uri=os.getenv('CORE_DB_URI')),  # type: ignore
            # memory_manager=VolatileMemoryManager(),
        ),
    )


@asynccontextmanager
async def manage_resources(app: FastAPI):  # noqa: ARG001
    agent_container = AgentContainer()
    await _initialize(agent_container)
    yield
    await _release(agent_container)


async def _initialize(agent_container: AgentContainer):
    agent_container.check_dependencies()
    qa_agent = agent_container.qa_agent()
    await qa_agent.initialize()

    agent_container.wire(
        modules=[
            'llm_chatbot_for_messengers.messenger.kakao.container',
        ]
    )


async def _release(agent_container: AgentContainer):
    qa_agent = agent_container.qa_agent()
    await qa_agent.shutdown()
    agent_container.unwire()
    agent_container.reset_singletons()


@inject
def _get_qa_agent(agent: QAAgent = Provide[AgentContainer.qa_agent]) -> QAAgent:
    """
    Returns:
        QAAgent: Fully initialized instance
    """
    return agent


def get_qa_agent() -> QAAgent:
    return _get_qa_agent()
