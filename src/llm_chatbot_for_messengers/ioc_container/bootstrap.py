from contextlib import asynccontextmanager

from fastapi import FastAPI

from llm_chatbot_for_messengers.ioc_container.container import AgentContainer, DaoContainer, MiddlewareContainer


@asynccontextmanager
async def manage_resources(app: FastAPI):  # noqa: ARG001
    agent_container = AgentContainer()
    middleware_container = MiddlewareContainer()
    dao_container = DaoContainer()

    await _initialize(agent_container, middleware_container, dao_container)
    yield
    await _release(agent_container, middleware_container, dao_container)


async def _initialize(
    agent_container: AgentContainer, middleware_container: MiddlewareContainer, dao_container: DaoContainer
):
    agent_container.check_dependencies()
    qa_agent = agent_container.qa_agent()
    await qa_agent.initialize()

    agent_container.wire(
        modules=[
            'llm_chatbot_for_messengers.ioc_container.container',
        ]
    )

    middleware_container.check_dependencies()
    middleware_container.wire(
        modules=[
            'llm_chatbot_for_messengers.ioc_container.container',
        ]
    )

    dao_container.check_dependencies()
    dao_container.wire(
        modules=[
            'llm_chatbot_for_messengers.ioc_container.container',
        ]
    )


async def _release(
    agent_container: AgentContainer, middleware_container: MiddlewareContainer, dao_container: DaoContainer
):
    qa_agent = agent_container.qa_agent()
    await qa_agent.shutdown()
    agent_container.unwire()
    agent_container.reset_singletons()

    middleware_container.unwire()
    middleware_container.reset_singletons()

    dao_container.unwire()
    dao_container.reset_singletons()
