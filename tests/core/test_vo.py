import pytest
from llm_chatbot_for_messengers.core.configuration import AgentConfig, AgentExtraConfig, LLMConfig, WorkflowNodeConfig
from llm_chatbot_for_messengers.core.output.memory import VolatileMemoryManager
from llm_chatbot_for_messengers.core.vo import UserId
from pydantic import ValidationError


@pytest.mark.parametrize(
    ('user_seq', 'user_id', 'expected'),
    [(1, 'test_id', 'ok'), (1, None, 'ok'), (None, 'test_id', 'ok'), (None, None, 'error')],
)
def test_user_id(user_seq: int, user_id: str, expected: str):
    match expected:
        case 'ok':
            assert UserId(user_seq=user_seq, user_id=user_id)
        case 'error':
            with pytest.raises(ValidationError):
                UserId(user_seq=user_seq, user_id=user_id)


def test_workflow_config():
    # given
    builder = AgentConfig.builder()
    manager = VolatileMemoryManager()
    # when
    workflow_config = (
        builder.add_node(node_name='answer_node', template_name='test-template', llm_config=LLMConfig())
        .add_fallback('fallback')
        .add_memory_manager(manager)
        .build()
    )
    # then
    assert workflow_config.node_configs.keys() == {'answer_node'}
    assert workflow_config.node_configs['answer_node'] == WorkflowNodeConfig(
        node_name='answer_node', template_name='test-template', llm_config=LLMConfig()
    )
    assert workflow_config.global_configs == AgentExtraConfig(fallback_message='fallback', memory_manager=manager)
