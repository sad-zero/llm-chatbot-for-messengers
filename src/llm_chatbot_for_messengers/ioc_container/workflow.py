from __future__ import annotations

import json
import logging
import re
from typing import Literal

import aiohttp
from bs4 import BeautifulSoup
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import BaseChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_core.runnables import Runnable, RunnablePassthrough
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import override

from llm_chatbot_for_messengers.domain.chatbot import ChatbotState, InnerState
from llm_chatbot_for_messengers.domain.specification import WorkflowSpecification

logger = logging.getLogger(__name__)

class ContextChatbotState(InnerState):
    question: str = Field(description='Question')
    context: str = Field(description='Context to answer the question', default='No context')
    messages: list[AnyMessage] = Field(description='Chat histories')
    answer: str | None = Field(description='Answer', default=None)

    @override
    def convert(self) -> ChatbotState:
        return ChatbotState(
            question=self.question,
            answer=self.answer,
        )
    @override
    def transfer(self) -> InnerState:
        raise NotImplementedError

    def get_formatted_messages(self) -> str:
        """Format messages as tuples
        Returns:
            str: [
                ("system", ...),
                ("human", ...),
                ("ai", ...),
            ]
        """
        result: list[tuple[Literal['system', 'human', 'ai'], str | list[str | dict]]] = []
        for message in self.messages:
            if isinstance(message, SystemMessage):
                result.append(('system', message.content))
            elif isinstance(message, HumanMessage):
                result.append(('human', message.content))
            elif isinstance(message, AIMessage):
                result.append(('ai', message.content))
            else:
                err_msg = f'Invalid message type: {message}'
                raise TypeError(err_msg)
        return json.dumps(result, indent=4, ensure_ascii=False)


class SummaryNodeDocument(BaseModel):
    title: str | None
    chunks: list[str]


class WebSummaryState(InnerState):
    question: str = Field(description='Question')
    url: str = Field(description='Website url')
    html_document: str | None = Field(description='HTML document crawled at url', default=None)
    document: SummaryNodeDocument | None = Field(description='Parsed document', default=None)
    error_message: str | None = Field(description='Error message', default=None)
    summary: str | None = Field(description="Agent's summary", default=None)

    @override
    def convert(self) -> ChatbotState:
        raise NotImplementedError
    @override
    def transfer(self) -> ContextChatbotState:
        return ContextChatbotState(
            question=self.question,
            context=self.summary,
        )

class SummaryNodeResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    summary: str = Field(description="Summary node's output")

class AnswerNodeResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    sentences: list[str] = Field(description="Answer node's output")

async def start_node_func(*, prompts: dict[str, BaseChatPromptTemplate] | None, llm: BaseChatModel | None, state: ChatbotState) -> ContextChatbotState | WebSummaryState:  # noqa: RUF029, ARG001
    if (url := extract_url(state.question)) is None:
        return ContextChatbotState(question=state.question)
    return WebSummaryState(question=state.question, url=url)

def extract_url(question: str | None) -> str | None:
    """Extract url from question

    Args:
        question (str | None): raw question

    Returns:
        str | None: url or None
    """
    if not isinstance(question, str):
        err_msg: str = f'question should be str: {question}'
        logger.warning(err_msg)
        return None
    pattern: str = '(https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*))'
    if (matched := re.search(pattern, question, re.DOTALL | re.IGNORECASE)) is not None:
        return matched.group(1)
    return None

async def start_node_conditional_func(*, prompts: dict[str, BaseChatPromptTemplate] | None, llm: BaseChatModel | None, state: ContextChatbotState | WebSummaryState) -> list[str]:
    if isinstance(state, ContextChatbotState):
        return ['context_qa_node']
    if isinstance(state, WebSummaryState):
        return ['web_summary_node']

async def context_qa_node_func(*, prompts: dict[str, BaseChatPromptTemplate] | None, llm: BaseChatModel | None, state: ContextChatbotState) -> ChatbotState:
    template: BaseChatPromptTemplate = prompts['kakao_v3'].partial(
        context=state.context, messages=state.get_formatted_messages(),
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

    return ChatbotState(
        question=state.question,
        answer='\n\n'.join(answer.sentences),
        inner_state=ContextChatbotState(
            question=state.question,
            answer=f'{answer!r}',
            context=state.context,
            messages=state.messages,
        )
    )


async def web_summary_node_func(session: aiohttp.ClientSession, *, prompts: dict[str, BaseChatPromptTemplate] | None, llm: BaseChatModel | None, state: WebSummaryState) -> WebSummaryState:  # noqa: ARG001
    """Crawl url and return HTML.
    Args:
        state   (WebSummaryState): {
            "url": ...,
        }
        session (aiohttp.ClientSession)
    Returns:
        success (dict): {
            "html_document": ...,
        }
        error   (dict): {

            "error_message": ...,
        }
    """
    try:
        async with session.get(str(state.url)) as resp:
            if not resp.ok:
                err_msg = f'Please check the given url: {state.url}'
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


async def web_summary_node_parse_func(*, prompts: dict[str, BaseChatPromptTemplate] | None, llm: BaseChatModel | None, state: WebSummaryState) -> WebSummaryState:
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
    if state.error_message is not None:
        return state
    soup = BeautifulSoup(state.html_document, 'lxml')  # type: ignore
    if (title_tag := soup.find('title')) is not None:
        title = title_tag.get_text(strip=True)
    content = content_tag.get_text(strip=True) if (content_tag := soup.find('body')) else 'Empty document'

    # Cut maximum tokens
    chunk_size: int = 700
    middle_idx: int = len(content) // 2

    first_chunk = content[:chunk_size]
    second_chunk = content[middle_idx - chunk_size // 2 : middle_idx + chunk_size // 2]
    last_chunk = content[-chunk_size:]
    chunks: list[str] = [first_chunk, second_chunk, last_chunk]

    document: SummaryNodeDocument = {
        'title': title,
        'chunks': chunks,
    }
    return {
        'document': document,
    }


async def web_summary_node_final_func(*, prompts: dict[str, BaseChatPromptTemplate] | None, llm: BaseChatModel | None, state: WebSummaryState) -> ContextChatbotState:
    """Summarize document.
    Args:
        state    (WebSummaryState): {
            "url": ...,
            "document": ...,
        }
        llm        (BaseChatModel): LLM for answer node
        template_name (str | None): Prompt Template name
    Returns:
        dict: {
            "summary": ...,
        }
    """
    if state.error_message is not None:
        return ContextChatbotState(
            context=state.error_message,
        )
    template = prompts['kakao_v2']
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
    return ContextChatbotState(context=answer.summary)

def get_workflow_spec() -> WorkflowSpecification:
    # start_node_spec = WorkflowNodeSpecification(
    #    initial_schema=ChatbotState,
    #    final_schemas=(ContextChatbotState, WebSummaryState,),
    #    name='start_node',
    #    func=start_node_func,
    #    conditional_edges=True,
    #    conditional_func=start_node_conditional_func,
    # )
    # TODO: impl
    raise NotImplementedError
