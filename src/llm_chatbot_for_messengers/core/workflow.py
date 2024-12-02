from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from llm_chatbot_for_messengers.core.custom_langgraph import PydanticStateGraph, Workflow
from llm_chatbot_for_messengers.core.template import get_template
from llm_chatbot_for_messengers.core.vo import AnswerNodeResponse, QAState

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


async def answer_node(state: QAState, llm: BaseChatModel, template_name: str | None = None) -> QAState:
    """Make a final answer.

    Args:
        state            (QAState): {
            "question": ...,
            "messages": ...,
        }
        llm        (BaseChatModel): LLM for answer node
        template_name (str | None): Prompt Template name
    Returns:
        QAState: {
            "answer": ...
        }
    """
    if llm is None:
        error_msg: str = 'LLM is not passed!'
        raise RuntimeError(error_msg)

    template = get_template(node_name='answer_node', template_name=template_name).partial(
        messages=state.get_formatted_messages()
    )
    try:
        chain: Runnable = (
            {'question': RunnablePassthrough()}
            | template
            | llm.with_structured_output(AnswerNodeResponse, method='json_schema')
        )
    except NotImplementedError:
        chain = (
            {'question': RunnablePassthrough()}
            | template
            | llm
            | PydanticOutputParser(pydantic_object=AnswerNodeResponse)
        )

    answer: AnswerNodeResponse = await chain.ainvoke(state.question)
    return QAState.put_answer(answer=answer.answer)


def get_question_answer_workflow(
    answer_node_llm: BaseChatModel | None = None, answer_node_template_name: str | None = None
) -> Workflow[QAState]:
    """
    Args:
        answer_node_llm (BaseChatModel | None): LLM for answer node
        answer_node_template_name (str | None): Prompt template for answer node
    Returns:
        Workflow: Question Answer Workflow
    """
    if answer_node_llm is None:
        answer_node_llm = ChatOpenAI(model='gpt-4o-2024-08-06', temperature=0.52, top_p=0.7, max_tokens=200)
    answer_node_with_llm = partial(answer_node, llm=answer_node_llm, template_name=answer_node_template_name)
    builder = PydanticStateGraph(QAState)
    builder.add_node('answer_node', answer_node_with_llm)
    builder.set_entry_point('answer_node')
    builder.set_finish_point('answer_node')
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)
