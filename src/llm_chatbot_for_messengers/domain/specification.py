"""
Domain Specification
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque
from enum import Enum, unique
from typing import Callable, Generic, Literal, Self, TypeVar, TypeVarTuple, Union

from langchain.prompts import BaseChatPromptTemplate  # noqa: TCH002
from langchain_core.language_models import BaseChatModel  # noqa: TCH002
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import override

from llm_chatbot_for_messengers.domain.chatbot import LLM, Chatbot, ChatbotState, Memory, Prompt, Workflow, WorkflowNode
from llm_chatbot_for_messengers.domain.messenger import MessengerId

logger = logging.getLogger(__name__)

T = TypeVar('T')


class Specification(ABC, BaseModel, Generic[T]):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @abstractmethod
    async def is_satisfied_by(self, t: T) -> bool:
        pass

    def and_(self, spec: Specification) -> Specification:
        return AndSpecification[T](one=self, other=spec)


class AndSpecification(Specification[T], Generic[T]):
    one: Specification[T]
    other: Specification[T]

    @override
    async def is_satisfied_by(self, t: T) -> bool:
        return (await self.one.is_satisfied_by(t)) and (await self.other.is_satisfied_by(t))


class ChatbotSpecification(Specification[Chatbot]):
    workflow_spec: WorkflowSpecification = Field(description="Workflow's specification")
    memory_spec: MemorySpecification = Field(description="Memory's specification")
    prompt_specs: list[PromptSpecification] = Field(description="Prompt's specification")
    timeout: int = Field(description='Configure max latency seconds', gt=0)
    fallback_message: str = Field(
        description='Answer this when time is over.', default='Too complex to answer in time.'
    )

    @override
    async def is_satisfied_by(self, t: Chatbot) -> bool:
        if not await self.workflow_spec.is_satisfied_by(t.workflow):
            err_msg = f"Workflow doesn't fulfill spec: {self.workflow_spec:r}, workflow: {t.workflow:r}"
            return _fail_validation(err_msg)
        if not await self.memory_spec.is_satisfied_by(t.memory):
            err_msg = f"Memory doesn't fulfill spec: {self.memory_spec:r}, memory: {t.memory:r}"
            return _fail_validation(err_msg)
        for prompt_spec, prompt in zip(self.prompt_specs, t.prompts):
            if not await prompt_spec.is_satisfied_by(prompt):
                err_msg = f"Prompt doesn't fulfill spec: {prompt_spec:r}, prompt: {prompt:r}"
                return _fail_validation(err_msg)
        if self.timeout != t.timeout:
            err_msg = f"Timeout doesn't fulfill spec: {self.timeout:r}, timeout: {t.timeout:r}"
            return _fail_validation(err_msg)
        if self.fallback_message != t.fallback_message:
            err_msg = (
                f"Fallback message doesn't fulfill spec: {self.fallback_message:r}, timeout: {t.fallback_message:r}"
            )
            return _fail_validation(err_msg)
        return True


class WorkflowSpecification(Specification[Workflow]):
    start_node_spec: WorkflowNodeSpecification[ChatbotState, BaseModel] = Field(  # type: ignore
        description="Workflow's start node specification"
    )
    end_node_spec: WorkflowNodeSpecification[BaseModel, ChatbotState] = Field(
        description="Workflow's end node specification"
    )
    # graph: CompiledGraph = Field(description="Configure workflow's graph")

    @model_validator(mode='after')
    def verify_reachable(self) -> Self:
        if not self.__is_reachable(self.start_node_spec, self.end_node_spec):
            err_msg: str = f"Start node doesn't reach end node: {self:r}"
            raise ValueError(err_msg)
        return self

    @override
    async def is_satisfied_by(self, t: Workflow) -> bool:
        if not await self.start_node_spec.is_satisfied_by(t.start_node):
            err_msg = f"Start node doesn't fulfill spec: {self.start_node_spec:r}, type: {t.start_node:r}"
            return _fail_validation(err_msg)
        if not await self.end_node_spec.is_satisfied_by(t.end_node):
            err_msg = f"End node doesn't fulfill spec: {self.end_node_spec:r}, type: {t.end_node:r}"
            return _fail_validation(err_msg)
        if self.end_node_spec.children_spec:
            err_msg = f'End node has children: {self.end_node_spec:r}, type: {t.end_node:r}'
            return _fail_validation(err_msg)
        return True

    @classmethod
    def __is_reachable(cls, start: WorkflowNodeSpecification, end: WorkflowNodeSpecification) -> bool:
        queue: deque[WorkflowNodeSpecification] = deque([start])
        visited = set()
        while queue:
            node = queue.popleft()
            if node.name == end.name:
                return True
            if node.name not in visited:
                visited.add(node.name)
                queue.extend(child for child in node.children_spec)
        return False


InitialState = TypeVar('InitialState', bound=BaseModel)
FinalStates = TypeVarTuple('FinalStates')


class WorkflowNodeSpecification(
    Specification[WorkflowNode[InitialState, *FinalStates]], Generic[InitialState, *FinalStates]
):
    initial_schema: type[InitialState] = Field(description="Configure node's initial schema")  # type: ignore
    final_schemas: tuple[*FinalStates, ...] = Field(description="Configure node's final schemas")  # type: ignore
    name: str = Field(description="Configure node's name")
    func: Callable[  # type: ignore
        [dict[str, BaseChatPromptTemplate] | None, BaseChatModel | None, InitialState], Union[*FinalStates]
    ] = Field(description="Configure node's function")
    llm: LLM | None = Field(description="Configure node's llm", default=None)
    children_spec: list[WorkflowNodeSpecification] = Field(
        description="Configure children's specification", default_factory=list
    )
    conditional_edges: bool = Field(
        description='Configure whether children are connected conditionally or not.', default=False
    )
    conditional_func: Callable[[*FinalStates], list[str]] | None = Field(
        description='Configure conditional rules.', default=None
    )

    @model_validator(mode='after')
    def verify_conditional(self) -> Self:
        if self.conditional_edges and (self.conditional_func is None):
            err_msg = "There are conditional edges but conditional func doesn't set"
            raise ValueError(err_msg)
        return self

    @override
    async def is_satisfied_by(self, t: WorkflowNode[InitialState, *FinalStates]) -> bool:
        return await self.travel(t)

    async def travel(self, start: WorkflowNode[InitialState, *FinalStates]) -> bool:
        """Validate whole nodes
        Returns:
            bool: Validation Result
        """
        visited: set[str] = set()
        queue: deque[tuple[WorkflowNodeSpecification, WorkflowNode]] = deque([(self, start)])
        while queue:
            spec, node = queue.popleft()
            if not await spec.__is_satisfied_by(node):  # noqa: SLF001
                return False
            if node.name not in visited:
                visited.add(node.name)
                queue.extend(zip(spec.children_spec, node.children))
        return True

    async def __is_satisfied_by(self, t: WorkflowNode[InitialState, *FinalStates]) -> bool:
        if self.initial_schema != t.initial_schema:
            err_msg = (
                f"Initial schema doesn't fulfill spec: {self.initial_schema:r}, initial schema: {t.initial_schema:r}"
            )
            return _fail_validation(err_msg)
        if not any(issubclass(final_schema, BaseModel) for final_schema in self.final_schemas):  # type: ignore
            err_msg = f"Final schemas doesn't BaseModel: {self.final_schemas:r}"
            return _fail_validation(err_msg)
        if self.final_schemas != t.final_schemas:
            err_msg = f"Final schemas doesn't fulfill spec: {self.final_schemas:r}, final schema: {t.final_schemas:r}"
            return _fail_validation(err_msg)
        if self.name != t.name:
            err_msg = f"Name doesn't fulfill spec: {self.name:r}, name: {t.name:r}"
            return _fail_validation(err_msg)
        if self.conditional_edges != t.conditional_edges:
            err_msg = f"Conditional edges doesn't fulfill spec: {self.conditional_edges:r}, conditional edges: {t.conditional_edges:r}"
            return _fail_validation(err_msg)
        if self.conditional_func != t.conditional_func:
            err_msg = f"Conditional func doesn't fulfill spec: {self.conditional_func:r}, conditional func: {t.conditional_func:r}"
            return _fail_validation(err_msg)
        return True

    def add_children(self, *children: WorkflowNodeSpecification) -> Self:
        if not any(isinstance(child, WorkflowNodeSpecification) for child in children):
            err_msg: str = f"Children doesn't WorkflowNodeSpecification: {children}"
            raise TypeError(err_msg)
        self.children_spec.extend(children)
        return self


class MemorySpecification(Specification[Memory]):
    type_: Literal['volatile', 'persistant'] = Field(description="Configure memory's type")
    conn_uri: str | None = Field(description="Configure memory's connection uri", default=None)
    conn_pool_size: int | None = Field(description="Configure memory's connection pool size", default=None)

    @model_validator(mode='after')
    def validate_persistant_info(self) -> Self:
        match self.type_:
            case 'persistant':
                if self.conn_uri is None or not self.conn_uri.startswith('postgresql://'):
                    err_msg: str = f'Conn uri is invalid: postgresql://xxx, conn_uri: {self.conn_uri:r}'
                    raise ValueError(err_msg)
                if self.conn_pool_size is None or not self.conn_pool_size > 0:
                    err_msg = f'Conn pool size is invalid: > 0, conn_pool_size: {self.conn_pool_size:r}'
                    raise ValueError(err_msg)
            case _:
                pass
        return self

    @override
    async def is_satisfied_by(self, t: Memory) -> bool:
        if self.type_ != t.type_:
            err_msg = f"Type doesn't fulfill spec: {self.type_:r}, type: {t.type_:r}"
            return _fail_validation(err_msg)
        if self.conn_uri != t.conn_uri:
            err_msg = f"Conn uri doesn't fulfill spec: {self.conn_uri:r}, type: {t.conn_uri:r}"
            return _fail_validation(err_msg)
        if self.conn_pool_size != t.conn_pool_size:
            err_msg = f"Conn pool size doesn't fulfill spec: {self.conn_pool_size:r}, type: {t.conn_pool_size:r}"
            return _fail_validation(err_msg)
        return True


class PromptSpecification(Specification[Prompt]):
    node: str = Field(description="Configure workflow's node name")
    name: str = Field(description="Configure prompt's name")

    @override
    async def is_satisfied_by(self, t: Prompt) -> bool:
        if self.node != t.node:
            err_msg: str = f"Node doesn't fulfill spec: {self.node:r}, workflow: {t.node:r}"
            return _fail_validation(err_msg)
        if self.name != t.name:
            err_msg = f"Name doesn't fulfill spec: {self.name:r}, workflow: {t.name:r}"
            return _fail_validation(err_msg)
        return True


class MessengerSpecification:
    # TODO: impl
    pass


class ChatSpecification:
    # TODO: impl
    pass


class TracingSpecification:
    # TODO: impl
    pass


def _fail_validation(err_msg: str) -> bool:
    logger.error(err_msg)
    return False


@unique
class MessengerIdEnum(Enum):
    KAKAO = MessengerId(messenger_seq=1, messenger_id='kakao')
