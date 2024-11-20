"""
Domain Specification
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

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
