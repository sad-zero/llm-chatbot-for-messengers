from typing import Generator

from llm_chatbot_for_messengers.core.agent import QAAgent, QAAgentImpl
from llm_chatbot_for_messengers.core.vo import LLMConfig, WorkflowNodeConfig


def get_qa_agent() -> Generator[QAAgent, None, None]:
    """
    Returns:
        QAAgent: Fully initialized instance
    """
    try:
        agent = QAAgentImpl.get_instance(
            workflow_configs={
                'answer_node': WorkflowNodeConfig(
                    node_name='answer_node',
                    template_name='kakao_v1',
                    llm_config=LLMConfig(model='gpt-4o-2024-08-06', temperature=0.52, max_tokens=200),
                )
            },
            fallback_message='미안해용. ㅠㅠ 질문이 너무 어려워용..',
        )
        yield agent
    finally:
        ...
