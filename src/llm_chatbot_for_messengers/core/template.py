from __future__ import annotations

from langchain.prompts import BasePromptTemplate, ChatPromptTemplate


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
        case _:
            error_msg: str = f'There are no templates for {node_name}'
            raise RuntimeError(error_msg)


def _get_template_of_answer_node(template_name: str | None) -> BasePromptTemplate:
    """Get answer node's Prompt Template

    Args:
        template_name (str | None): Template Name

    Returns:
        BasePromptTemplate: Prompt Template
    Raises:
        RuntimeError: Not Found
    """
    match template_name:
        case None:
            return ChatPromptTemplate.from_messages([
                (
                    'system',
                    "Please act as a kindly chatbot.\nCould you answer human's question?.\nPlease answer in *KOREAN*",
                ),
                ('human', '{question}'),
            ])
        case _:
            error_msg: str = f'There are no templates for {template_name}'
            raise RuntimeError(error_msg)
