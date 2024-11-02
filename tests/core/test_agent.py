from __future__ import annotations

import pytest
from llm_chatbot_for_messengers.core.agent import QAAgent
from llm_chatbot_for_messengers.core.vo import WorkflowNodeConfig


@pytest.mark.parametrize(
    ('workflow_configs', 'expected'),
    [
        ({'valid-node': WorkflowNodeConfig(node_name='valid-node')}, 'ok'),
        ({'invalid-node': WorkflowNodeConfig(node_name='valid-node')}, 'error'),
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
