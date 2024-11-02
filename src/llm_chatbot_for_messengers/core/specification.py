"""
Domain Specification
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_chatbot_for_messengers.core.vo import WorkflowNodeConfig


def check_workflow_configs(workflow_configs: dict[str, WorkflowNodeConfig]) -> dict[str, WorkflowNodeConfig]:
    if not all(key == value.node_name for key, value in workflow_configs.items()):
        err_msg: str = f'{workflow_configs} should equal key = value.node_name'
        raise RuntimeError(err_msg)
    return workflow_configs
