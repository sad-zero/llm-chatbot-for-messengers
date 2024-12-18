from __future__ import annotations

import pytest
from llm_chatbot_for_messengers.domain.factory import ChatbotImpl
from llm_chatbot_for_messengers.domain.messenger import User, UserId
from llm_chatbot_for_messengers.domain.specification import AgentConfig, AgentExtraConfig, WorkflowNodeConfig
from llm_chatbot_for_messengers.infra.repository.memory import VolatileMemoryManager


@pytest.mark.parametrize(
    ('workflow_configs', 'global_configs', 'expected'),
    [
        (
            {'answer_node': WorkflowNodeConfig(node_name='answer_node')},
            AgentExtraConfig(fallback_message='Fallback message'),
            'ok',
        ),
        (
            {'answer_node': WorkflowNodeConfig(node_name='answer_node')},
            AgentExtraConfig(fallback_message='Fallback message', memory_manager=VolatileMemoryManager()),
            'ok',
        ),
        (
            {'invalid_node': WorkflowNodeConfig(node_name='answer_node')},
            AgentExtraConfig(fallback_message='Fallback message'),
            'error',
        ),
        (
            {'answer_node': WorkflowNodeConfig(node_name='invalid_node')},
            AgentExtraConfig(fallback_message='Fallback message'),
            'error',
        ),
    ],
)
def test_create_qa_agent(
    workflow_configs: dict[str, WorkflowNodeConfig], global_configs: AgentExtraConfig, expected: str
):
    if expected == 'ok':
        # when
        assert ChatbotImpl(config=AgentConfig(node_configs=workflow_configs, global_configs=global_configs)) is not None
        # then
    else:
        # when
        # then
        with pytest.raises(RuntimeError):
            assert ChatbotImpl(config=AgentConfig(node_configs=workflow_configs, global_configs=global_configs))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('user', 'question', 'expected'),
    [
        (User(user_id=UserId(user_seq=1)), 'How do I make vlog?', 'ok'),
        (User(user_id=UserId(user_seq=1)), None, 'error'),
        (None, 'How do I make vlog?', 'error'),
        (None, None, 'error'),
    ],
)
async def test_qa_agent_ask(user, question, expected, fake_agent):
    match expected:
        case 'ok':
            answer: str = await fake_agent.ask(user=user, question=question)
            assert answer is not None
        case 'error':
            with pytest.raises(TypeError):
                await fake_agent.ask(user=user, question=question)
