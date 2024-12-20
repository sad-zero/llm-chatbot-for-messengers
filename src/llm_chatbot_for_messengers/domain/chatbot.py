from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC
from collections import deque
from enum import Enum, unique
from functools import partial
from typing import (
    Annotated,
    Any,
    Callable,
    Generic,
    Literal,
    Self,
    TypeVar,
    TypeVarTuple,
    Union,
)

from langchain.prompts import BaseChatPromptTemplate  # noqa: TCH002
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel  # noqa: TCH002
from langchain_core.messages import AnyMessage  # noqa: TCH002
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver  # noqa: TCH002
from langgraph.graph import StateGraph, add_messages
from langgraph.graph.graph import CompiledGraph  # noqa: TCH002
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class ChatbotState(BaseModel):
    model_config = ConfigDict(frozen=True)

    question: str = Field(description="User's question")
    answer: str | None = Field(description="Chatbot's answer", default=None)


class Prompt(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    node: str = Field(description="Configure workflow's node name")
    name: str = Field(description="Configure prompt's name")
    template: BaseChatPromptTemplate = Field(description="Configure prompt's template")


class Memory(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    type_: Literal['volatile', 'persistant'] = Field(description="Configure memory's type")
    conn_uri: str | None = Field(description="Configure memory's connection uri", default=None)
    conn_pool_size: int | None = Field(description="Configure memory's connection pool size", default=None)

    memory: BaseCheckpointSaver = Field(description='Configure memory')


class Chatbot(ABC, BaseModel):
    workflow: Workflow = Field(description='Configure senarios')
    memory: Memory = Field(description='Store chat histories')
    prompts: list[Prompt] = Field(description="Configure LLM's prompts")
    timeout: int = Field(description='Configure max latency seconds', gt=0)
    fallback_message: str = Field(
        description='Answer this when time is over.', default='Too complex to answer in time.'
    )

    async def answer(self, question: str, **kwargs: Any) -> ChatbotState:
        """Answer the question
        Args:
            question (str): Question
            **kwargs (Any): extra informations(Don't extract directly)
        """
        try:
            initial_state: ChatbotState = ChatbotState(question=question)
            final_state: ChatbotState = await asyncio.wait_for(
                self.workflow.execute(initial_state, **kwargs), self.timeout
            )
        except asyncio.CancelledError:
            err_info: dict = {'timeout': self.timeout, 'question': question, 'kwargs': kwargs}
            logger.exception(json.dumps(err_info, ensure_ascii=False))
            return ChatbotState(question=question, answer=self.fallback_message)
        else:
            return final_state


class Workflow(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    start_node: WorkflowNode[ChatbotState, BaseModel] = Field(description="Workflow's start node")  # type: ignore
    end_node: WorkflowNode[BaseModel, ChatbotState] = Field(description="Workflow's end node")  # type: ignore
    graph: CompiledGraph = Field(description='Executable graph')

    async def execute(self, initial: ChatbotState, **kwargs) -> ChatbotState:
        """Execute workflow
        Args:
            initial (StateSchema): Initial state
            **kwargs        (Any): extra informations(Don't extract directly)
        Returns:
            StateSchema          : Final state
        """
        graph_response: dict = await self.graph.ainvoke(initial, **kwargs)
        return ChatbotState.model_validate(graph_response, strict=True)

    @classmethod
    def _build_llm(cls, llm_config: LLM) -> BaseChatModel:
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
    def _graph_builder(cls, state_schema: type[ChatbotState]) -> StateGraph:
        return StateGraph(state_schema=state_schema)


InitialState = TypeVar('InitialState', bound=BaseModel)
FinalStates = TypeVarTuple('FinalStates')


class WorkflowNode(BaseModel, Generic[InitialState, *FinalStates]):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    initial_schema: type[InitialState] = Field(description='Type holder')  # type: ignore
    final_schemas: tuple[*FinalStates, ...] = Field(description='Type holder')  # type: ignore
    name: str = Field(description="Node's name")
    func: Callable[  # type: ignore
        [dict[str, BaseChatPromptTemplate] | None, BaseChatModel | None, InitialState], Union[*FinalStates]
    ] = Field(description="Node's function")
    llm: LLM | None = Field(description="Node's llm", default=None)
    children: list[WorkflowNode] = Field(description="Node's children", default_factory=list)

    def travel(self) -> list[WorkflowNode]:
        """BFS travel
        Returns:
            list[WorkflowNode]: Nodes from Self
        """
        result: list[WorkflowNode] = []
        visited: set[str] = set()
        queue: deque[WorkflowNode] = deque([self, *self.children])
        while queue:
            node = queue.popleft()
            result.append(node)  # Add to path
            if node.name not in visited:
                visited.add(node.name)
                queue.extend(node.children)
        return result

    def get_executable_node(
        self, prompts: dict[str, BaseChatPromptTemplate] | None, llm: BaseChatModel | None
    ) -> Callable[InitialState, Union[*FinalStates]]:  # type: ignore
        """Return executable node function
        Args:
            prompt (dict | None): {
                "prompt_name": prompt,
            }
            llm (BaseChatModel | None): used in function
        Returns:
            Callable[NodeStateSchema, Union[*NodeStateSchemas]]: Executable function
        """
        return partial(self.func, prompts, llm)

    def add_children(self, *children: WorkflowNode) -> Self:
        if not any(isinstance(child, WorkflowNode) for child in children):
            err_msg: str = f"Children doesn't WorkflowNode: {children}"
            raise TypeError(err_msg)
        self.children.extend(children)
        return self


@unique
class LLMProvider(Enum):
    OPENAI: str = 'OPENAI'

    def __str__(self):
        return self.value


class LLM(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: LLMProvider = Field(description='LLM Provider', default=LLMProvider.OPENAI)
    model: str = Field(description='LLM Name', default='gpt-4o-2024-08-06')
    temperature: float = Field(description='LLM Temperature', default=0.52)
    top_p: float = Field(description='LLM Top-p', default=0.95)
    max_tokens: int = Field(description='Maximum completion tokens', default=500)
    extra_configs: dict[str, Any] = Field(
        description='Extra configurations for provider, model, etc', default_factory=dict
    )


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


# class QAWorkflow(Workflow[QAState]):
#     @classmethod
#     @override
#     def get_instance(cls, config: dict[str, WorkflowNodeConfig], memory: MemoryType | None = None) -> Self:
#         if (answer_node_config := config.get('answer_node')) is None:
#             answer_node_config = WorkflowNodeConfig(node_name='answer_node')
#         answer_node_llm = cls._build_llm(llm_config=answer_node_config.llm_config)
#         answer_node_with_llm = partial(
#             cls.__answer_node, llm=answer_node_llm, template_name=answer_node_config.template_name
#         )

#         graph = (
#             cls._graph_builder(state_schema=QAState)
#             .add_node('answer_node', answer_node_with_llm)
#             .set_entry_point('answer_node')
#             .set_finish_point('answer_node')
#             .compile(checkpointer=memory)
#         )
#         return cls(compiled_graph=graph, state_schema=QAState)

#     @staticmethod
#     async def __answer_node(state: QAState, llm: BaseChatModel, template_name: str | None = None) -> QAState:
#         """Make a final answer.

#         Args:
#             state            (QAState): {
#                 "question": ...,
#                 "context": ...,
#                 "messages": ...,
#             }
#             llm        (BaseChatModel): LLM for answer node
#             template_name (str | None): Prompt Template name
#         Returns:
#             QAState: {
#                 "answer": ...
#             }
#         """
#         if llm is None:
#             error_msg: str = 'LLM is not passed!'
#             raise WorkflowError(error_msg)

#         template = get_template(node_name='answer_node', template_name=template_name).partial(
#             context=state.context, messages=state.get_formatted_messages()
#         )
#         try:
#             chain: Runnable = (
#                 {'question': RunnablePassthrough()}
#                 | template
#                 | llm.with_structured_output(AnswerNodeResponse, method='json_schema')
#             )
#         except NotImplementedError:
#             chain = (
#                 {'question': RunnablePassthrough()}
#                 | template
#                 | llm
#                 | PydanticOutputParser(pydantic_object=AnswerNodeResponse)
#             )

#         answer: AnswerNodeResponse = await chain.ainvoke(state.question)
#         return QAState.put_answer(answer='\n\n'.join(answer.sentences))


# # Define Workflow States


# class SummaryNodeDocument(TypedDict):
#     title: str | None
#     chunks: list[str]


# class WebSummaryState(BaseModel):
#     url: str = Field(description='Website url')
#     html_document: str | None = Field(description='HTML document crawled at url', default=None)
#     document: SummaryNodeDocument | None = Field(description='Parsed document', default=None)
#     error_message: str | None = Field(description='Error message', default=None)
#     summary: str | None = Field(description="Agent's summary", default=None)

#     @classmethod
#     def initialize(cls, url: str) -> Self:
#         return cls(url=url)  # type: ignore


# class SummaryNodeResponse(BaseModel):
#     model_config = ConfigDict(frozen=True)

#     summary: str = Field(description="Summary node's output")


# class WebSummaryWorkflow(Workflow[WebSummaryState]):
#     @classmethod
#     @override
#     def get_instance(cls, config: dict[str, WorkflowNodeConfig], memory: MemoryType | None = None) -> Self:
#         if (summary_node_config := config.get('summary_node')) is None:
#             summary_node_config = WorkflowNodeConfig(node_name='summary_node')
#         summary_node_llm = cls._build_llm(llm_config=summary_node_config.llm_config)
#         summary_node_with_llm = partial(
#             cls.__summary_node,
#             llm=summary_node_llm,
#             template_name=summary_node_config.template_name,
#         )
#         # TODO: close session
#         session = aiohttp.ClientSession()
#         crawl_url_node = partial(cls.__crawl_url_node, session=session)
#         graph = (
#             cls._graph_builder(state_schema=WebSummaryState)
#             .add_node('crawl_url_node', crawl_url_node)
#             .add_node('parse_node', cls.__parse_node)
#             .add_node('summary_node', summary_node_with_llm)
#             .add_edge(START, 'crawl_url_node')
#             .add_conditional_edges('crawl_url_node', cls.__control, ['parse_node', END])
#             .add_edge('parse_node', 'summary_node')
#             .add_edge('summary_node', END)
#             .compile(checkpointer=memory)
#         )
#         return cls(compiled_graph=graph, state_schema=WebSummaryState)

#     @staticmethod
#     async def __control(state: WebSummaryState) -> list[str]:
#         """Control workflow's conditions
#         Args:
#             state (WebSummaryState): {
#                 "url": ...,
#                 "html_document": ...,
#                 "error_message": ...,
#             }
#         Returns:
#             list[node_name]: next_node_names
#         """
#         if state.error_message is not None:
#             return [END]
#         return ['parse_node']

#     @staticmethod
#     async def __crawl_url_node(state: WebSummaryState, session: aiohttp.ClientSession) -> dict:
#         """Crawl url and return HTML.
#         Args:
#             state   (WebSummaryState): {
#                 "url": ...,
#             }
#             session (aiohttp.ClientSession)
#         Returns:
#             success (dict): {
#                 "html_document": ...,
#             }
#             error   (dict): {

#                 "error_message": ...,
#             }
#         """
#         if not isinstance(session, aiohttp.ClientSession):
#             err_msg: str = f'session should be aiohttp.ClientSession type: {session:r}'
#             raise TypeError(err_msg)
#         try:
#             async with session.get(str(state.url)) as resp:
#                 if not resp.ok:
#                     err_msg = f'Please check the given url: {state.url}'
#                     resp_info: str = f"""
#                     URL: {state.url}
#                     HEADER: {resp.headers}
#                     BODY: {await resp.text()}
#                     """
#                     logger.warning(resp_info)
#                     return {
#                         'error_message': err_msg,
#                     }
#                 return {
#                     'html_document': await resp.text(),
#                 }
#         except aiohttp.ClientConnectorDNSError:
#             err_msg = f'Url cannot be crawled: {state.url}'
#             logger.exception(err_msg)
#             return {
#                 'error_message': err_msg,
#             }

#     @staticmethod
#     async def __parse_node(
#         state: WebSummaryState,
#     ) -> dict:
#         """Parse html_document
#         Args:
#             state (WebSummaryState): {
#                 "html_document": ...,
#             }
#         Returns:
#             dict: {
#                 "document": ...
#             }
#         """
#         soup = BeautifulSoup(state.html_document, 'lxml')  # type: ignore
#         if (title_tag := soup.find('title')) is not None:
#             title = title_tag.get_text().strip()
#         if (content_tag := soup.find('body')) is not None:
#             content = content_tag.get_text(strip=True)
#         else:
#             content = 'Empty document'

#         # Cut maximum tokens
#         chunk_size: int = 700
#         middle_idx: int = len(content) // 2

#         first_chunk = content[:chunk_size]
#         second_chunk = content[middle_idx - chunk_size // 2 : middle_idx + chunk_size // 2]
#         last_chunk = content[-chunk_size:]
#         chunks: list[str] = [first_chunk, second_chunk, last_chunk]

#         document: SummaryNodeDocument = {
#             'title': title,
#             'chunks': chunks,
#         }
#         return {
#             'document': document,
#         }

#     @staticmethod
#     async def __summary_node(state: WebSummaryState, llm: BaseChatModel, template_name: str | None = None) -> dict:
#         """Summarize document.
#         Args:
#             state    (WebSummaryState): {
#                 "url": ...,
#                 "document": ...,
#             }
#             llm        (BaseChatModel): LLM for answer node
#             template_name (str | None): Prompt Template name
#         Returns:
#             dict: {
#                 "summary": ...,
#             }
#         """
#         if llm is None:
#             error_msg: str = 'LLM is not passed!'
#             raise WorkflowError(error_msg)

#         template = get_template(node_name='summary_node', template_name=template_name)
#         try:
#             chain: Runnable = (
#                 {'document': RunnablePassthrough()}
#                 | template
#                 | llm.with_structured_output(SummaryNodeResponse, method='json_schema')
#             )
#         except NotImplementedError:
#             chain = (
#                 {'document': RunnablePassthrough()}
#                 | template
#                 | llm
#                 | PydanticOutputParser(pydantic_object=SummaryNodeResponse)
#             )

#         answer: SummaryNodeResponse = await chain.ainvoke(state.document)
#         return {
#             'summary': answer.summary,
#         }


# class QAWithWebSummaryState(BaseModel):
#     question: str | None = Field(description="User's question", default=None)
#     context: str = Field(description='Context to answer the question', default='No context')
#     messages: Annotated[list[AnyMessage], add_messages] = Field(description='Chat histories', default_factory=list)
#     answer: str | None = Field(description="Agent's answer", default=None)


# class QAWithWebSummaryWorkflow(Workflow[QAWithWebSummaryState]):
#     @classmethod
#     @override
#     def get_instance(cls, config: dict[str, WorkflowNodeConfig], memory: MemoryType | None = None) -> Self:
#         qa_workflow: QAWorkflow = QAWorkflow.get_instance(config=config, memory=memory)
#         web_summary_workflow: WebSummaryWorkflow = WebSummaryWorkflow.get_instance(config=config, memory=memory)

#         qa_node = partial(cls.__qa_node, workflow=qa_workflow)
#         web_summary_node = partial(cls.__web_summary_node, workflow=web_summary_workflow)
#         graph = (
#             cls._graph_builder(QAWithWebSummaryState)
#             .add_node('qa_node', qa_node)
#             .add_node('web_summary_node', web_summary_node)
#             .add_conditional_edges(START, cls.__route_workflows, ['qa_node', 'web_summary_node'])
#             .add_edge('qa_node', END)
#             .add_edge('web_summary_node', 'qa_node')
#             .compile(checkpointer=memory)
#         )
#         return cls(compiled_graph=graph, state_schema=QAWithWebSummaryState)

#     @staticmethod
#     async def __route_workflows(state: QAWithWebSummaryState) -> list[str]:
#         """Route workflows

#         Args:
#             state (QAWithWebSummaryState): {
#                 "question": ...,
#             }

#         Returns:
#             list[str]: Next nodes
#         """
#         if QAWithWebSummaryWorkflow.__extract_url(state.question) is None:
#             return ['qa_node']
#         return ['web_summary_node']

#     @staticmethod
#     async def __qa_node(state: QAWithWebSummaryState, workflow: QAWorkflow) -> dict:
#         """Invoke QAWorkflow and return result.

#         Args:
#             state (QAWithWebSummaryState): {
#                 "question": ...,
#                 "context": ...,
#             }
#             workflow (QAWorkflow): QAWorkflowImpl

#         Returns:
#             dict: {
#                 "answer": ...,
#             }
#         """
#         if not isinstance(workflow, QAWorkflow):
#             err_msg: str = f'workflow should be QAWorkflow: {workflow!r}'
#             raise TypeError(err_msg)

#         qa_state: QAState = QAState(question=state.question, context=state.context, messages=state.messages)  # type: ignore
#         response: QAState = await workflow.ainvoke(qa_state)
#         return {
#             'answer': response.answer,
#         }

#     @staticmethod
#     async def __web_summary_node(state: QAWithWebSummaryState, workflow: WebSummaryWorkflow) -> dict:
#         """Invoke WebSummaryWorkflow and return result.

#         Args:
#             state (QAWithWebSummaryState): {
#                 "question": ...,
#             }
#             workflow (WebSummaryWorkflow): WebSummaryWorkflowImpl

#         Returns:
#             dict: {
#                 "context": ...,
#             }
#         """
#         if not isinstance(workflow, WebSummaryWorkflow):
#             err_msg: str = f'workflow should be WebSummaryWorkflow: {workflow!r}'
#             raise TypeError(err_msg)

#         url = QAWithWebSummaryWorkflow.__extract_url(state.question)
#         web_summary_state: WebSummaryState = WebSummaryState(url=url)  # type: ignore
#         response: WebSummaryState = await workflow.ainvoke(web_summary_state)

#         return {'context': response.summary or response.error_message}

#     @staticmethod
#     def __extract_url(question: str | None) -> str | None:
#         """Extract url from question

#         Args:
#             question (str | None): raw question

#         Returns:
#             str | None: url or None
#         """
#         if not isinstance(question, str):
#             err_msg: str = f'question should be str: {question}'
#             logger.warning(err_msg)
#             return None
#         pattern: str = '(https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*))'
#         if (matched := re.search(pattern, question, re.DOTALL | re.IGNORECASE)) is not None:
#             return matched.group(1)
#         return None
