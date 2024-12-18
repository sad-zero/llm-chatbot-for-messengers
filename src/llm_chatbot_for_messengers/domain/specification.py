"""
Domain Specification
"""

from __future__ import annotations

import asyncio
import inspect
import time
from enum import Enum, unique
from functools import partial, wraps
from threading import Thread
from typing import Annotated, Any, Callable, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from llm_chatbot_for_messengers.domain.error import SpecificationError
from llm_chatbot_for_messengers.domain.messenger import MessengerId


def check_timeout(func: Callable[..., Any], *, timeout: int) -> Callable[..., Any]:
    """Check agent/node timeout
    Args:
        func (Callable[..., Any]): Target function
        timeout             (int): Timeout Seconds
    Returns:
        Callable[..., Any]: Traced target function
    """
    if not isinstance(timeout, int):
        err_msg: str = 'timeout should be integer value(seconds)'
        raise TypeError(err_msg)
    if timeout <= 0:
        err_msg = 'timeout should be greater than 0'
        raise ValueError(err_msg)

    if func is None:
        return partial(check_timeout, timeout=timeout)

    if not inspect.isfunction(func):
        err_msg = 'func should be function'
        raise TypeError(err_msg)

    err_msg_template: str = 'timeout: {elapsed: .2f} is longer than {timeout}'

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        def _func_in_thread(*args, **kwargs) -> None:
            placeholder = kwargs.pop('placeholder')
            result: Any = func(*args, **kwargs)
            placeholder.append(result)

        placeholder: list[Any] = []
        if 'placeholder' in inspect.signature(func).parameters:
            err_msg = 'function should not have `placeholder` parameter'
            raise KeyError(err_msg)
        t = Thread(target=_func_in_thread, args=args, kwargs=kwargs | {'placeholder': placeholder})
        s: float = time.time()
        t.start()
        t.join(timeout=timeout)
        e: float = time.time()
        elapsed: float = e - s
        if elapsed >= timeout:
            err_msg = err_msg_template.format(elapsed=elapsed, timeout=timeout)
            raise SpecificationError(err_msg)
        return placeholder[0]

    @wraps(func)
    async def awrapper(*args, **kwargs) -> Any:
        s: float = time.time()
        try:
            async with asyncio.timeout(timeout):
                result: Any = await func(*args, **kwargs)
        except TimeoutError as err:
            e: float = time.time()
            elapsed: float = e - s
            if elapsed >= timeout:
                err_msg = err_msg_template.format(elapsed=elapsed, timeout=timeout)
                raise SpecificationError(err_msg) from err
        else:
            return result

    if inspect.iscoroutinefunction(func):
        return awrapper
    return wrapper


class ChatbotSpecification:
    # TODO: impl
    pass


class MessengerSpecification:
    # TODO: impl
    pass


class ChatSpecification:
    # TODO: impl
    pass


class TracingSpecification:
    # TODO: impl
    pass


def check_workflow_configs(workflow_configs: dict[str, WorkflowNodeConfig]) -> dict[str, WorkflowNodeConfig]:
    if not all(key == value.node_name for key, value in workflow_configs.items()):
        err_msg: str = f'{workflow_configs} should equal key = value.node_name'
        raise RuntimeError(err_msg)
    return workflow_configs


def check_necessary_nodes(
    node_names: str | list[str],
) -> Callable[[dict[str, WorkflowNodeConfig]], dict[str, WorkflowNodeConfig]]:
    if isinstance(node_names, str):
        node_names = [node_names]
    necessary_node_names: set[str] = set(node_names)

    def _check(workflow_configs: dict[str, WorkflowNodeConfig]) -> dict[str, WorkflowNodeConfig]:
        if set(workflow_configs.keys()).intersection(necessary_node_names) != necessary_node_names:
            err_msg: str = f'{workflow_configs} should contain {necessary_node_names}'
            raise RuntimeError(err_msg)
        return workflow_configs

    return _check


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


class WorkflowNodeConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    node_name: str = Field(description='Workflow node name')
    template_name: str | None = Field(description='Workflow node prompt template name', default=None)
    llm_config: LLMConfig = Field(description="Workflow node's LLM Config", default_factory=LLMConfig)


@unique
class MessengerIdEnum(Enum):
    KAKAO = MessengerId(messenger_seq=1, messenger_id='kakao')
