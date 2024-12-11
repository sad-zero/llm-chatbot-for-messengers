from __future__ import annotations

import json
from enum import Enum, unique
from typing import Annotated, Any, Literal, Self

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages import AnyMessage  # noqa: TCH002
from langgraph.graph import add_messages  # noqa: TCH002
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, HttpUrl, model_validator
from typing_extensions import TypedDict

from llm_chatbot_for_messengers.core.output.memory import MemoryManager  # noqa: TCH001
from llm_chatbot_for_messengers.core.specification import check_necessary_nodes, check_workflow_configs  # noqa: TCH001


class UserId(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_seq: int | None = Field(description="User's sequence", default=None)
    user_id: str | None = Field(description="User's unique string", default=None)

    @model_validator(mode='after')
    def check_id(self) -> Self:
        if (self.user_seq, self.user_id) == (None, None):
            err_msg: str = 'One of id or seq should exist.'
            raise ValueError(err_msg)
        return self


class MessengerId(BaseModel):
    model_config = ConfigDict(frozen=True)

    messenger_seq: int | None = Field(description="Messenger's sequence", default=None)
    messenger_id: str | None = Field(description="Messenger's unique string", default=None)

    @model_validator(mode='after')
    def check_id(self) -> Self:
        if (self.messenger_seq, self.messenger_id) == (None, None):
            err_msg: str = 'One of id or seq should exist.'
            raise ValueError(err_msg)
        return self


@unique
class MessengerIdEnum(Enum):
    KAKAO = MessengerId(messenger_seq=1, messenger_id='kakao')


class QAState(BaseModel):
    question: str | None = Field(description="User's question", default=None)
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


class WebSummaryState(BaseModel):
    url: HttpUrl = Field(description='Website url')
    html_document: str | None = Field(description='HTML document crawled at url', default=None)
    document: SummaryNodeDocument | None = Field(description='Parsed document', default=None)
    error_message: str | None = Field(description='Error message', default=None)
    summary: str | None = Field(description="Agent's summary", default=None)

    @classmethod
    def initialize(cls, url: str) -> Self:
        return cls(url=url)  # type: ignore


@unique
class LLMProvider(Enum):
    OPENAI: str = 'OPENAI'

    def __str__(self):
        return self.value


class LLMConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: LLMProvider = Field(description='LLM Provider', default=LLMProvider.OPENAI)
    model: str = Field(description='LLM Name', default='gpt-4o-2024-08-06')
    temperature: float = Field(description='LLM Temperature', default=0.52)
    top_p: float = Field(description='LLM Top-p', default=0.95)
    max_tokens: int = Field(description='Maximum completion tokens', default=500)
    extra_configs: dict[str, Any] = Field(
        description='Extra configurations for provider, model, etc', default_factory=dict
    )


class WorkflowNodeConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    node_name: str = Field(description='Workflow node name')
    template_name: str | None = Field(description='Workflow node prompt template name', default=None)
    llm_config: LLMConfig = Field(description="Workflow node's LLM Config", default_factory=LLMConfig)


class AgentExtraConfig(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    fallback_message: str = Field(description='Fallback message is returned when normal flows fail')

    memory_manager: MemoryManager | None = Field(
        description='Manager that control memories. None means stateless.', default=None
    )


class AgentConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    node_configs: Annotated[
        dict[str, WorkflowNodeConfig],
        AfterValidator(check_workflow_configs),
        AfterValidator(check_necessary_nodes('answer_node')),
    ] = Field(description='Node configurations. Key equals to node_name.')
    global_configs: AgentExtraConfig = Field(description='Global configurations')

    @classmethod
    def builder(cls) -> _Builder:
        return cls._Builder()

    class _Builder:
        def __init__(self):
            self.__node_configs = {}
            self.__global_configs = {}

        def add_node(self, node_name: str, template_name: str | None, llm_config: LLMConfig | None = None) -> Self:
            self.__node_configs[node_name] = WorkflowNodeConfig(
                node_name=node_name, template_name=template_name, llm_config=llm_config or LLMConfig()
            )
            return self

        def add_fallback(self, fallback_message: str) -> Self:
            self.__global_configs['fallback'] = fallback_message
            return self

        def add_memory_manager(self, memory_manager: MemoryManager) -> Self:
            self.__global_configs['memory_manager'] = memory_manager
            return self

        def build(self) -> AgentConfig:
            return AgentConfig(
                node_configs=self.__node_configs,
                global_configs=AgentExtraConfig(
                    fallback_message=self.__global_configs['fallback'],
                    memory_manager=self.__global_configs['memory_manager'],
                ),
            )


class AnswerNodeResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    answer: str = Field(description="Answer node's output")


class SummaryNodeDocument(TypedDict):
    title: str | None
    content: str
    is_end: bool


class SummaryNodeResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    summary: str = Field(description="Summary node's output")
