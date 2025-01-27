from __future__ import annotations

from langchain.prompts import BasePromptTemplate, ChatPromptTemplate

from llm_chatbot_for_messengers.core.output.parser import get_parser


def get_template(node_name: str, template_name: str | None = None) -> BasePromptTemplate:
    """Get Prompt Template

    Args:
        node_name (str): Node to use template
        template_name (str | None): Detail template name. Defaults to None.
    Returns:
        BasePromptTemplate: Prompt Template
    Raises:
        RuntimeError: Not Found
    """
    match node_name:
        case 'answer_node':
            return _get_template_of_answer_node(template_name=template_name)
        case 'summary_node':
            return _get_template_of_summary_node(template_name=template_name)
        case _:
            error_msg: str = f'There are no templates for {node_name}'
            raise ValueError(error_msg)


def _get_template_of_answer_node(template_name: str | None) -> BasePromptTemplate:
    """Get answer node's Prompt Template
    Args:
        template_name (str | None): Template Name

    Returns:
        BasePromptTemplate: Prompt Template
    """
    match template_name:
        case str():
            return get_parser().parse_file(node_name='answer_node', template_name=template_name)
        case None:
            return ChatPromptTemplate.from_messages([
                ('system', 'Please act as a helpful question-answering agent.'),
                ('human', 'my question is {question}'),
            ])
        case _:
            err_msg: str = f'template_name({template_name}) should be str'
            raise TypeError(err_msg)


def _get_template_of_summary_node(template_name: str | None) -> BasePromptTemplate:
    """Get summary node's Prompt Template
    Args:
        template_name (str | None): Template Name

    Returns:
        BasePromptTemplate: Prompt Template
    """
    match template_name:
        case str():
            return get_parser().parse_file(node_name='summary_node', template_name=template_name)
        case None:
            return ChatPromptTemplate.from_messages([
                ('system', 'Please summrize the document to read quickly main contents.'),
                ('human', '{document}'),
            ])
        case _:
            err_msg: str = f'template_name({template_name}) should be str'
            raise TypeError(err_msg)
