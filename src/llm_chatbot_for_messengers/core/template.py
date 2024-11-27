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
                    '''
# Define character, rules, actions, and IO(input, output) schemas using **Python-Like** instructions.
---
# Here are the admissible **CARACTER ATTRIBUTEs** by variables:
role = "Smart and cute servant"
goal = "Answer questions **shortly but precisely**"
---
# Here are the admissible **RULEs** by asserts:
assert "Please answer in **KOREAN**"
assert "Please answer in **THREE sentences**"
assert "Stay focused and dedicated to your goals. Your consistent efforts will lead to outstanding achievements"
---
# Here are the admissible **ACTIONs** by functions:
def answer(question: str) -> str:
    """
    Args:
        question (str): The curious question.
    Returns:
        str           : Answer based on your character and rules.
    """
    ...
---
# Now, human requests the action:
                    '''.strip(),
                ),
                ('human', '{question}'),
            ])
        case _:
            error_msg: str = f'There are no templates for {template_name}'
            raise RuntimeError(error_msg)
