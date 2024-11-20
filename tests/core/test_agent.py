from __future__ import annotations

import pytest
from llm_chatbot_for_messengers.core.agent import QAAgent
from llm_chatbot_for_messengers.core.vo import LLMConfig, WorkflowNodeConfig


@pytest.mark.parametrize(
    ('workflow_configs', 'expected'),
    [
        ({'answer_node': WorkflowNodeConfig(node_name='answer_node')}, 'ok'),
        ({'invalid_node': WorkflowNodeConfig(node_name='answer_node')}, 'error'),
        ({'answer_node': WorkflowNodeConfig(node_name='invalid_node')}, 'error'),
    ],
)
def test_create_qa_agent(workflow_configs: dict[str, WorkflowNodeConfig], expected: str):
    # when
    if expected == 'ok':
        # then
        assert QAAgent(workflow_configs=workflow_configs) is not None
    else:
        # then
        with pytest.raises(RuntimeError):
            QAAgent(workflow_configs=workflow_configs)


def test_qa_agent_cached_workflow():
    # given
    workflow_configs = {'answer_node': WorkflowNodeConfig(node_name='answer_node', llm_config=LLMConfig())}
    agent = QAAgent(workflow_configs=workflow_configs)
    # when
    workflow1 = agent.workflow
    workflow2 = agent.workflow
    # then
    assert id(workflow1) == id(workflow2)
