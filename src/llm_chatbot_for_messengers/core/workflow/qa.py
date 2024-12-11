from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING, Self

import aiohttp
import tiktoken
from bs4 import BeautifulSoup
from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough
from langgraph.graph import END, START
from typing_extensions import override

from llm_chatbot_for_messengers.core.error import WorkflowError
from llm_chatbot_for_messengers.core.output.template import get_template
from llm_chatbot_for_messengers.core.vo import (
    AnswerNodeResponse,
    QAState,
    SummaryNodeDocument,
    SummaryNodeResponse,
    WebSummaryState,
    WorkflowNodeConfig,
)
from llm_chatbot_for_messengers.core.workflow.base import Workflow

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from llm_chatbot_for_messengers.core.output.memory import MemoryType

logger = logging.getLogger(__name__)


class QAWorkflow(Workflow[QAState]):
    @classmethod
    @override
    def get_instance(cls, config: dict[str, WorkflowNodeConfig], memory: MemoryType | None = None) -> Self:
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
            raise WorkflowError(error_msg)

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


class WebSummaryWorkflow(Workflow[WebSummaryState]):
    @classmethod
    @override
    def get_instance(cls, config: dict[str, WorkflowNodeConfig], memory: MemoryType | None = None) -> Self:
        if (summary_node_config := config.get('summary_node')) is None:
            summary_node_config = WorkflowNodeConfig(node_name='summary_node')
        summary_node_llm = cls._build_llm(llm_config=summary_node_config.llm_config)
        summary_node_with_llm = partial(
            cls.__summary_node,
            llm=summary_node_llm,
            template_name=summary_node_config.template_name,
        )
        graph = (
            cls._graph_builder(state_schema=WebSummaryState)
            .add_node('crawl_url_node', cls.__crawl_url_node)
            .add_node('parse_node', cls.__parse_node)
            .add_node('summary_node', summary_node_with_llm)
            .add_edge(START, 'crawl_url_node')
            .add_conditional_edges('crawl_url_node', cls.__control, ['parse_node', END])
            .add_edge('parse_node', 'summary_node')
            .add_edge('summary_node', END)
            .compile(checkpointer=memory)
        )
        return cls(compiled_graph=graph, state_schema=WebSummaryState)

    @staticmethod
    async def __control(state: WebSummaryState) -> list[str]:
        """Control workflow's conditions
        Args:
            state (WebSummaryState): {
                "url": ...,
                "html_document": ...,
                "error_message": ...,
            }
        Returns:
            list[node_name]: next_node_names
        """
        if state.error_message is not None:
            return [END]
        return ['parse_node']

    @staticmethod
    async def __crawl_url_node(state: WebSummaryState) -> dict:
        """Crawl url and return HTML.
        Args:
            state   (WebSummaryState): {
                "url": ...,
            }
        Returns:
            success (dict): {
                "html_document": ...,
            }
            error   (dict): {

                "error_message": ...,
            }
        """
        try:
            async with aiohttp.ClientSession() as session, session.get(str(state.url)) as resp:
                if not resp.ok:
                    err_msg: str = f'Please check the given url: {state.url}'
                    resp_info: str = f"""
                    URL: {state.url}
                    HEADER: {resp.headers}
                    BODY: {await resp.text()}
                    """
                    logger.warning(resp_info)
                    return {
                        'error_message': err_msg,
                    }
                return {
                    'html_document': await resp.text(),
                }
        except aiohttp.ClientConnectorDNSError:
            err_msg = f'Url cannot be crawled: {state.url}'
            logger.exception(err_msg)
            return {
                'error_message': err_msg,
            }

    @staticmethod
    async def __parse_node(
        state: WebSummaryState,
    ) -> dict:
        """Parse html_document
        Args:
            state (WebSummaryState): {
                "html_document": ...,
            }
        Returns:
            dict: {
                "document": ...
            }
        """
        soup = BeautifulSoup(state.html_document, 'html.parser')  # type: ignore
        if (title_tag := soup.find('title')) is not None:
            title = title_tag.get_text().strip()
        if (content_tag := soup.find('body')) is not None:
            content = content_tag.get_text(strip=True)
        else:
            content = 'Empty document'

        # Cut maximum tokens
        encoder = tiktoken.get_encoding('o200k_base')  # gpt-4o, gpt-4o-mini tokenizer
        max_tokens = 20_000
        encoded_content: list[int] = encoder.encode(content)
        is_end: bool = len(encoded_content) <= max_tokens
        content = encoder.decode(encoded_content[:max_tokens])

        document: SummaryNodeDocument = {
            'title': title,
            'content': content,
            'is_end': is_end,
        }
        return {
            'document': document,
        }

    @staticmethod
    async def __summary_node(state: WebSummaryState, llm: BaseChatModel, template_name: str | None = None) -> dict:
        """Summarize document.
        Args:
            state    (WebSummaryState): {
                "url": ...,
                "html_document": ...,
            }
            llm        (BaseChatModel): LLM for answer node
            template_name (str | None): Prompt Template name
        Returns:
            dict: {
                "summary": ...,
            }
        """
        if llm is None:
            error_msg: str = 'LLM is not passed!'
            raise WorkflowError(error_msg)

        template = get_template(node_name='summary_node', template_name=template_name)
        try:
            chain: Runnable = (
                {'document': RunnablePassthrough()}
                | template
                | llm.with_structured_output(SummaryNodeResponse, method='json_schema')
            )
        except NotImplementedError:
            chain = (
                {'document': RunnablePassthrough()}
                | template
                | llm
                | PydanticOutputParser(pydantic_object=SummaryNodeResponse)
            )

        answer: SummaryNodeResponse = await chain.ainvoke(state.document)
        return {
            'summary': answer.summary,
        }
