from typing import Generator

from llm_chatbot_for_messengers.core.agent import QAAgent, QAAgentImpl
from llm_chatbot_for_messengers.core.vo import LLMConfig, WorkflowNodeConfig


def get_qa_agent() -> Generator[QAAgent, None, None]:
    """
    Returns:
        QAAgent: Fully initialized instance
    """
    try:
        agent = QAAgentImpl(
            workflow_configs={'answer_node': WorkflowNodeConfig(node_name='answer_node', llm_config=LLMConfig())}
        )
        yield agent
    finally:
        ...
