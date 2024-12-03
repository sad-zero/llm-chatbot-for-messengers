from __future__ import annotations

import pytest
from llm_chatbot_for_messengers.core.entity.agent import QAAgentImpl
from llm_chatbot_for_messengers.core.entity.user import User
from llm_chatbot_for_messengers.core.vo import LLMConfig, UserId, WorkflowNodeConfig


@pytest.mark.parametrize(
    ('workflow_configs', 'fallback_message', 'expected'),
    [
        ({'answer_node': WorkflowNodeConfig(node_name='answer_node')}, 'Fallback message', 'ok'),
        ({'invalid_node': WorkflowNodeConfig(node_name='answer_node')}, 'Fallback message', 'error'),
        ({'answer_node': WorkflowNodeConfig(node_name='invalid_node')}, 'Fallback message', 'error'),
    ],
)
def test_create_qa_agent(workflow_configs: dict[str, WorkflowNodeConfig], fallback_message: str, expected: str):
    # when
    if expected == 'ok':
        # then
        assert QAAgentImpl(workflow_configs=workflow_configs, fallback_message=fallback_message) is not None
    else:
        # then
        with pytest.raises(RuntimeError):
            QAAgentImpl(workflow_configs=workflow_configs, fallback_message=fallback_message)


def test_qa_agent_cached_workflow():
    # given
    workflow_configs = {'answer_node': WorkflowNodeConfig(node_name='answer_node', llm_config=LLMConfig())}
    agent = QAAgentImpl(workflow_configs=workflow_configs, fallback_message='Fallback message')
    # when
    workflow1 = agent.workflow
    workflow2 = agent.workflow
    # then
    assert id(workflow1) == id(workflow2)


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
