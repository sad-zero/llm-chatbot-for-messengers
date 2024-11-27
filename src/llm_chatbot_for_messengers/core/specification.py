"""
Domain Specification
"""

from __future__ import annotations

import inspect
import time
from functools import partial, wraps
from typing import TYPE_CHECKING, Any, Callable

from llm_chatbot_for_messengers.core.error import SpecificationError

if TYPE_CHECKING:
    from llm_chatbot_for_messengers.core.vo import WorkflowNodeConfig


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

    err_msg_template: str = '{elapsed: .2f} is longer than {timeout}'

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        s: float = time.time()
        result: Any = func(*args, **kwargs)
        e: float = time.time()
        elapsed: float = e - s
        if elapsed >= timeout:
            err_msg = err_msg_template.format(elapsed=elapsed, timeout=timeout)
            raise SpecificationError(err_msg)
        return result

    @wraps(func)
    async def awrapper(*args, **kwargs) -> Any:
        s: float = time.time()
        result: Any = await func(*args, **kwargs)
        e: float = time.time()
        elapsed: float = e - s
        if elapsed >= timeout:
            err_msg = err_msg_template.format(elapsed=elapsed, timeout=timeout)
            raise SpecificationError(err_msg)
        return result

    if inspect.iscoroutinefunction(func):
        return awrapper
    return wrapper
