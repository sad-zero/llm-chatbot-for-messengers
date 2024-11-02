"""
Domain
"""

from __future__ import annotations

from typing import Annotated

from pydantic import AfterValidator, BaseModel, Field

from llm_chatbot_for_messengers.core.specification import check_workflow_configs  # noqa: TCH001
from llm_chatbot_for_messengers.core.vo import WorkflowNodeConfig  # noqa: TCH001


class QAAgent(BaseModel):
    workflow_configs: Annotated[dict[str, WorkflowNodeConfig], AfterValidator(check_workflow_configs)] = Field(
        description='Key equals to WorkflowNodeConfig.node_name'
    )
