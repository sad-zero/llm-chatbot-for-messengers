"""langgraph wrapper"""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

from langgraph.graph import StateGraph
from pydantic import BaseModel

if TYPE_CHECKING:
    from langgraph.graph.graph import CompiledGraph

StateSchema = TypeVar('StateSchema', bound=BaseModel)


class Workflow(Generic[StateSchema]):
    def __init__(self, compiled_graph: CompiledGraph, state_schema: type[StateSchema]):
        self.__compiled_graph = compiled_graph
        self.__state_schema = state_schema

    async def ainvoke(self, *args, **kwargs) -> StateSchema:
        response = await self.__compiled_graph.ainvoke(*args, **kwargs)
        result: StateSchema = self.__state_schema.model_validate(response)
        return result


class PydanticStateGraph(StateGraph, Generic[StateSchema]):
    def __init__(self, state_schema: type[StateSchema], *args, **kwargs):
        if not issubclass(state_schema, BaseModel):
            err_msg: str = f'state_schema should be pydantic model: {state_schema}'
            raise TypeError(err_msg)
        self.__pydantic_schema: type[BaseModel] = state_schema
        super().__init__(state_schema, *args, **kwargs)

    def compile(self, *args, **kwargs):
        compiled_graph = super().compile(*args, **kwargs)
        return Workflow(compiled_graph=compiled_graph, state_schema=self.__pydantic_schema)
