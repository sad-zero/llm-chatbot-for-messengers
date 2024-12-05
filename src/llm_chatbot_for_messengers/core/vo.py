from __future__ import annotations

import json
from enum import Enum, unique
from typing import Annotated, Any, Literal, Self

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages import AnyMessage  # noqa: TCH002
from langgraph.graph import add_messages  # noqa: TCH002
from pydantic import BaseModel, Field, model_validator

from llm_chatbot_for_messengers.core.output.memory import MemoryManager  # noqa: TCH001


class UserId(BaseModel):
    user_seq: int | None = Field(description="User's sequence", default=None)
    user_id: str | None = Field(description="User's unique string", default=None)

    @model_validator(mode='after')
    def check_id(self) -> Self:
        if (self.user_seq, self.user_id) == (None, None):
            err_msg: str = 'One of id or seq should exist.'
            raise ValueError(err_msg)
        return self


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


@unique
class LLMProvider(Enum):
    OPENAI: str = 'OPENAI'

    def __str__(self):
        return self.value


class LLMConfig(BaseModel):
    provider: LLMProvider = Field(description='LLM Provider', default=LLMProvider.OPENAI)
    model: str = Field(description='LLM Name', default='gpt-4o-2024-08-06')
    temperature: float = Field(description='LLM Temperature', default=0.52)
    top_p: float = Field(description='LLM Top-p', default=0.95)
    max_tokens: int = Field(description='Maximum completion tokens', default=500)
    extra_configs: dict[str, Any] = Field(
        description='Extra configurations for provider, model, etc', default_factory=dict
    )


class WorkflowNodeConfig(BaseModel):
    node_name: str = Field(description='Workflow node name')
    template_name: str | None = Field(description='Workflow node prompt template name', default=None)
    llm_config: LLMConfig | None = Field(description="Workflow node's LLM Config", default=None)


class WorkflowGlobalConfig(BaseModel):
    fallback_message: str = Field(description='Fallback message is returned when normal flows fail')

    memory_manager: MemoryManager | None = Field(
        description='Manager that control memories. None means stateless.', default=None
    )

    class Config:
        arbitrary_types_allowed = True


class AnswerNodeResponse(BaseModel):
    answer: str = Field(description="Answer node's output")
