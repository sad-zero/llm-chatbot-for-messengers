"""
Domain
"""

from __future__ import annotations

import logging
from functools import cached_property
from typing import TYPE_CHECKING, Annotated

from langchain_openai import ChatOpenAI
from pydantic import AfterValidator, BaseModel, Field

from llm_chatbot_for_messengers.core.specification import check_necessary_nodes, check_workflow_configs  # noqa: TCH001
from llm_chatbot_for_messengers.core.vo import LLMConfig, LLMProvider, WorkflowNodeConfig
from llm_chatbot_for_messengers.core.workflow import get_question_answer_workflow

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from llm_chatbot_for_messengers.core.workflow import Workflow

logger = logging.getLogger(__name__)


class QAAgent(BaseModel):
    workflow_configs: Annotated[
        dict[str, WorkflowNodeConfig],
        AfterValidator(check_workflow_configs),
        AfterValidator(check_necessary_nodes('answer_node')),
    ] = Field(description='Key equals to WorkflowNodeConfig.node_name')

    @cached_property
    def workflow(self) -> Workflow:
        answer_node_config: WorkflowNodeConfig = self.workflow_configs['answer_node']
        answer_node_llm: BaseChatModel | None = None
        if answer_node_config.llm_config is not None:
            answer_node_llm = self.__build_llm(llm_config=answer_node_config.llm_config)
        return get_question_answer_workflow(answer_node_llm=answer_node_llm)

    @staticmethod
    def __build_llm(llm_config: LLMConfig) -> BaseChatModel:
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
