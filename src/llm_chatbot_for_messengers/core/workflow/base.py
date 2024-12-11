from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, Self, TypeVar

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.graph import CompiledGraph
from pydantic import BaseModel

from llm_chatbot_for_messengers.core.vo import LLMConfig, LLMProvider, WorkflowNodeConfig

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from llm_chatbot_for_messengers.core.output.memory import MemoryType

StateSchema = TypeVar('StateSchema', bound=BaseModel)


class Workflow(ABC, Generic[StateSchema]):
    def __init__(self, compiled_graph: CompiledGraph, state_schema: type[StateSchema]):
        if not isinstance(compiled_graph, CompiledGraph):
            err_msg: str = f'compiled_graph should be CompiledGraph type: {compiled_graph}'
            raise TypeError(err_msg)
        if not issubclass(state_schema, BaseModel):
            err_msg = f'state_schema should be subtype of BaseModel: {state_schema}'
            raise TypeError(err_msg)

        self.__compiled_graph = compiled_graph
        self.__state_schema = state_schema

    async def ainvoke(self, *args, **kwargs) -> StateSchema:
        response = await self.__compiled_graph.ainvoke(*args, **kwargs)
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
