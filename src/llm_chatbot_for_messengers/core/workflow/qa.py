from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Self

from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough
from typing_extensions import override

from llm_chatbot_for_messengers.core.output.template import get_template
from llm_chatbot_for_messengers.core.vo import AnswerNodeResponse, QAState, WorkflowNodeConfig
from llm_chatbot_for_messengers.core.workflow.base import Workflow

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from llm_chatbot_for_messengers.core.output.memory import MemoryType


class QAWorkflow(Workflow[QAState]):
    @classmethod
    @override
    def get_instance(cls, config: dict[str, WorkflowNodeConfig], memory: MemoryType | None) -> Self:
        if (answer_node_config := config.get('answer_node')) is None:
            answer_node_config = WorkflowNodeConfig(node_name='answer_node')
        answer_node_llm = cls._build_llm(llm_config=answer_node_config.llm_config)
        answer_node_with_llm = partial(
            cls.__answer_node, llm=answer_node_llm, template_name=answer_node_config.template_name
        )

        graph = (
            cls._graph_builder(state_schema=QAState)
            .add_node('answer_node', answer_node_with_llm)
            .set_entry_point('answer_node')
            .set_finish_point('answer_node')
            .compile(checkpointer=memory)
        )
        return cls(compiled_graph=graph, state_schema=QAState)

    @staticmethod
    async def __answer_node(state: QAState, llm: BaseChatModel, template_name: str | None = None) -> QAState:
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
