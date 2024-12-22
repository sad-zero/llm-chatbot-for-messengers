import pytest
from llm_chatbot_for_messengers.domain.chatbot import ChatbotState
from llm_chatbot_for_messengers.domain.factory import ChatbotFactory
from llm_chatbot_for_messengers.domain.specification import (
    ChatbotSpecification,
    MemorySpecification,
    PromptSpecification,
    WorkflowNodeSpecification,
    WorkflowSpecification,
)


@pytest.mark.asyncio
async def test_chatbot_factory():
    # given
    chatbot_spec = ChatbotSpecification(
        workflow_spec=WorkflowSpecification(
            start_node_spec=WorkflowNodeSpecification[ChatbotState, ChatbotState](
                initial_schema=ChatbotState,
                final_schemas=(ChatbotState,),
                name='start',
                func=lambda x, y, z: (x, y, z),
                children_spec=[
                    WorkflowNodeSpecification[ChatbotState, ChatbotState](
                        initial_schema=ChatbotState,
                        final_schemas=(ChatbotState,),
                        name='end',
                        func=lambda x, y, z: (x, y, z),
                    )
                ],
            ),
            end_node_spec=WorkflowNodeSpecification[ChatbotState, ChatbotState](  # type: ignore
                initial_schema=ChatbotState,
                final_schemas=(ChatbotState,),
                name='end',
                func=lambda x, y, z: (x, y, z),
            ),
        ),
        memory_spec=MemorySpecification(type_='volatile'),
        prompt_specs=[
            PromptSpecification(
                node='start',
                name='test',
            )
        ],
        timeout=5,
    )
    chatbot_factory = ChatbotFactory()
    # when
    async with chatbot_factory.create_chatbot(chatbot_spec) as chatbot:
        # then
        assert await chatbot_spec.is_satisfied_by(chatbot)
