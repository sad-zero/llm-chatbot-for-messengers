from __future__ import annotations

from enum import Enum, unique
from typing import Any, NotRequired, Self, TypedDict

from pydantic import BaseModel, Field, model_validator


class UserId(BaseModel):
    user_seq: int | None = Field(description="User's sequence", default=None)
    user_id: str | None = Field(description="User's unique string", default=None)

    @model_validator(mode='after')
    def check_id(self) -> Self:
        if (self.user_seq, self.user_id) == (None, None):
            err_msg: str = 'One of id or seq should exist.'
            raise ValueError(err_msg)
        return self


class QAState(TypedDict):
    question: str
    answer: NotRequired[str]  # Filled by Agent


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


class AnswerNodeResponse(BaseModel):
    answer: str = Field(description="Answer node's output")
