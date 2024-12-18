from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from functools import partial
from typing import TYPE_CHECKING, Annotated, Generic, Literal, Self, TypeVar

import aiohttp
from bs4 import BeautifulSoup
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages import AnyMessage  # noqa: TCH002
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.graph.graph import CompiledGraph
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict, override

from llm_chatbot_for_messengers.domain.error import SpecificationError, WorkflowError
from llm_chatbot_for_messengers.domain.messenger import User
from llm_chatbot_for_messengers.domain.specification import LLMConfig, LLMProvider, WorkflowNodeConfig, check_timeout
from llm_chatbot_for_messengers.infra.chatbot import get_template

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from llm_chatbot_for_messengers.domain.repository import MemoryType

logger = logging.getLogger(__name__)


class Chatbot(ABC):
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the agent.
        - Acquire memory
        - etc
        """

    @abstractmethod
    async def shutdown(self) -> None:
        """
        Shutdown the agent.
        - Release memory
        - etc
        """

    async def ask(self, user: User, question: str, timeout: int | None = None) -> str:
        """Ask question

        Args:
            user          (User): User Information
            question       (str): User's question
            timeout (int | None): Timeout seconds
        Returns:
            str: Agent's answer
        """
        if not isinstance(user, User) or not isinstance(question, str):
            err_msg: str = f'Invalid arguments: user: {user}, question: {question}'
            raise TypeError(err_msg)
        try:
            if timeout is None:
                return await self._ask(user=user, question=question)
            traced = check_timeout(self.__class__._ask, timeout=timeout)  # noqa: SLF001
            return await traced(self=self, user=user, question=question)
        except SpecificationError:
            logger.exception('')
            return await self._fallback(user=user, question=question)

    @abstractmethod
    async def _ask(self, user: User, question: str) -> str:
        """Answer user's question
        Args:
            user    (User): User Information
            question (str): User's question
        Returns:
            str: Agent's answer
        """

    @abstractmethod
    async def _fallback(self, user: User, question: str) -> str:
        """Fallback message based on user's question(**Return immidiately**)
        Args:
            user    (User): User Information
            question (str): User's question
        Returns:
            str: Static fallback message
        """


StateSchema = TypeVar('StateSchema', bound=BaseModel)


class Workflow(ABC, Generic[StateSchema]):
    def __init__(self, compiled_graph: CompiledGraph, state_schema: type[StateSchema]):
        if not isinstance(compiled_graph, CompiledGraph):
            err_msg: str = f'compiled_graph should be CompiledGraph type: {compiled_graph}'
            raise TypeError(err_msg)
        if not issubclass(state_schema, BaseModel):
            err_msg = f'state_schema should be subtype of BaseModel: {state_schema}'
            raise TypeError(err_msg)

        self._compiled_graph = compiled_graph
        self.__state_schema = state_schema

    async def ainvoke(self, *args, **kwargs) -> StateSchema:
        response = await self._compiled_graph.ainvoke(*args, **kwargs)
        result: StateSchema = self.__state_schema.model_validate(response)
        return result

    @classmethod
    @abstractmethod
    def get_instance(cls, config: dict[str, WorkflowNodeConfig], memory: MemoryType | None = None) -> Self:
        pass

    @classmethod
    def _build_llm(cls, llm_config: LLMConfig) -> BaseChatModel:
        match llm_config.provider:
            case LLMProvider.OPENAI:
                model = ChatOpenAI(
                    model=llm_config.model,
                    temperature=llm_config.temperature,
                    top_p=llm_config.top_p,
                    max_tokens=llm_config.max_tokens,
                    **llm_config.extra_configs,
                )
            case _:
                err_msg: str = f'Cannot support {llm_config.provider} provider now.'
                raise RuntimeError(err_msg)
        return model

    @classmethod
    def _graph_builder(cls, state_schema: type[StateSchema]) -> StateGraph:
        return StateGraph(state_schema=state_schema)


class QAState(BaseModel):
    question: str | None = Field(description="User's question", default=None)
    context: str = Field(description='Context to answer the question', default='No context')
    answer: str | None = Field(description="Agent's answer", default=None)
    messages: Annotated[list[AnyMessage], add_messages] = Field(description='Chat histories', default_factory=list)

    @classmethod
    def put_question(cls, question: str) -> Self:
        return cls(
            question=question,
            messages=[HumanMessage(question)],
        )

    @classmethod
    def put_answer(cls, answer: str) -> Self:
        return cls(
            answer=answer,
            messages=[AIMessage(answer)],
        )

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


# Define LLM Response formats


class AnswerNodeResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    sentences: list[str] = Field(description="Answer node's output")


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
                "context": ...,
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
            context=state.context, messages=state.get_formatted_messages()
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
        return QAState.put_answer(answer='\n\n'.join(answer.sentences))


# Define Workflow States


class SummaryNodeDocument(TypedDict):
    title: str | None
    chunks: list[str]


class WebSummaryState(BaseModel):
    url: str = Field(description='Website url')
    html_document: str | None = Field(description='HTML document crawled at url', default=None)
    document: SummaryNodeDocument | None = Field(description='Parsed document', default=None)
    error_message: str | None = Field(description='Error message', default=None)
    summary: str | None = Field(description="Agent's summary", default=None)

    @classmethod
    def initialize(cls, url: str) -> Self:
        return cls(url=url)  # type: ignore


class SummaryNodeResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    summary: str = Field(description="Summary node's output")


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
        # TODO: close session
        session = aiohttp.ClientSession()
        crawl_url_node = partial(cls.__crawl_url_node, session=session)
        graph = (
            cls._graph_builder(state_schema=WebSummaryState)
            .add_node('crawl_url_node', crawl_url_node)
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
    async def __crawl_url_node(state: WebSummaryState, session: aiohttp.ClientSession) -> dict:
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
        if not isinstance(session, aiohttp.ClientSession):
            err_msg: str = f'session should be aiohttp.ClientSession type: {session:r}'
            raise TypeError(err_msg)
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
        soup = BeautifulSoup(state.html_document, 'lxml')  # type: ignore
        if (title_tag := soup.find('title')) is not None:
            title = title_tag.get_text().strip()
        if (content_tag := soup.find('body')) is not None:
            content = content_tag.get_text(strip=True)
        else:
            content = 'Empty document'

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

    @staticmethod
    async def __summary_node(state: WebSummaryState, llm: BaseChatModel, template_name: str | None = None) -> dict:
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


class QAWithWebSummaryState(BaseModel):
    question: str | None = Field(description="User's question", default=None)
    context: str = Field(description='Context to answer the question', default='No context')
    messages: Annotated[list[AnyMessage], add_messages] = Field(description='Chat histories', default_factory=list)
    answer: str | None = Field(description="Agent's answer", default=None)


class QAWithWebSummaryWorkflow(Workflow[QAWithWebSummaryState]):
    @classmethod
    @override
    def get_instance(cls, config: dict[str, WorkflowNodeConfig], memory: MemoryType | None = None) -> Self:
        qa_workflow: QAWorkflow = QAWorkflow.get_instance(config=config, memory=memory)
        web_summary_workflow: WebSummaryWorkflow = WebSummaryWorkflow.get_instance(config=config, memory=memory)

        qa_node = partial(cls.__qa_node, workflow=qa_workflow)
        web_summary_node = partial(cls.__web_summary_node, workflow=web_summary_workflow)
        graph = (
            cls._graph_builder(QAWithWebSummaryState)
            .add_node('qa_node', qa_node)
            .add_node('web_summary_node', web_summary_node)
            .add_conditional_edges(START, cls.__route_workflows, ['qa_node', 'web_summary_node'])
            .add_edge('qa_node', END)
            .add_edge('web_summary_node', 'qa_node')
            .compile(checkpointer=memory)
        )
        return cls(compiled_graph=graph, state_schema=QAWithWebSummaryState)

    @staticmethod
    async def __route_workflows(state: QAWithWebSummaryState) -> list[str]:
        """Route workflows

        Args:
            state (QAWithWebSummaryState): {
                "question": ...,
            }

        Returns:
            list[str]: Next nodes
        """
        if QAWithWebSummaryWorkflow.__extract_url(state.question) is None:
            return ['qa_node']
        return ['web_summary_node']

    @staticmethod
    async def __qa_node(state: QAWithWebSummaryState, workflow: QAWorkflow) -> dict:
        """Invoke QAWorkflow and return result.

        Args:
            state (QAWithWebSummaryState): {
                "question": ...,
                "context": ...,
            }
            workflow (QAWorkflow): QAWorkflowImpl

        Returns:
            dict: {
                "answer": ...,
            }
        """
        if not isinstance(workflow, QAWorkflow):
            err_msg: str = f'workflow should be QAWorkflow: {workflow!r}'
            raise TypeError(err_msg)

        qa_state: QAState = QAState(question=state.question, context=state.context, messages=state.messages)  # type: ignore
        response: QAState = await workflow.ainvoke(qa_state)
        return {
            'answer': response.answer,
        }

    @staticmethod
    async def __web_summary_node(state: QAWithWebSummaryState, workflow: WebSummaryWorkflow) -> dict:
        """Invoke WebSummaryWorkflow and return result.

        Args:
            state (QAWithWebSummaryState): {
                "question": ...,
            }
            workflow (WebSummaryWorkflow): WebSummaryWorkflowImpl

        Returns:
            dict: {
                "context": ...,
            }
        """
        if not isinstance(workflow, WebSummaryWorkflow):
            err_msg: str = f'workflow should be WebSummaryWorkflow: {workflow!r}'
            raise TypeError(err_msg)

        url = QAWithWebSummaryWorkflow.__extract_url(state.question)
        web_summary_state: WebSummaryState = WebSummaryState(url=url)  # type: ignore
        response: WebSummaryState = await workflow.ainvoke(web_summary_state)

        return {'context': response.summary or response.error_message}

    @staticmethod
    def __extract_url(question: str | None) -> str | None:
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
