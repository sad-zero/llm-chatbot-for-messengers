from __future__ import annotations

from enum import Enum, unique
from typing import Any, TypedDict

from pydantic import BaseModel, Field


class QAState(TypedDict):
    question: str
    answer: str


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
    llm_config: LLMConfig | None = Field(description="Workflow node's LLM Config", default=None)
