from __future__ import annotations

from enum import Enum, unique
from typing import Annotated, Any, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from llm_chatbot_for_messengers.domain.specification import (  # noqa: TCH001
    check_necessary_nodes,
    check_workflow_configs,
)


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

    memory_manager: Any = Field(description='Manager that control memories. None means stateless.', default=None)


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

        def add_memory_manager(self, memory_manager: Any) -> Self:
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
