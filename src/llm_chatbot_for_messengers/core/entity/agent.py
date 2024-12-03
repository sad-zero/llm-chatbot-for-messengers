"""
Domain
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, Annotated, Self, cast

from langchain_openai import ChatOpenAI
from pydantic import AfterValidator, BaseModel, Field
from typing_extensions import override

from llm_chatbot_for_messengers.core.entity.user import User
from llm_chatbot_for_messengers.core.error import SpecificationError
from llm_chatbot_for_messengers.core.specification import (
    check_necessary_nodes,
    check_timeout,
    check_workflow_configs,
)
from llm_chatbot_for_messengers.core.vo import LLMConfig, LLMProvider, QAState, WorkflowNodeConfig
from llm_chatbot_for_messengers.core.workflow import get_question_answer_workflow

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from llm_chatbot_for_messengers.core.custom_langgraph import Workflow

logger = logging.getLogger(__name__)


class QAAgent(ABC):
    __instance: Self | None = None

    @classmethod
    def get_instance(cls, *args, **kwargs) -> Self:
        if cls.__instance is None:
            cls.__instance = cls(*args, **kwargs)
        return cls.__instance

    async def ask(self, user: User, question: str, timeout: int | None = None) -> str:
        """Ask question

        Args:
            user          (User): User Information
            question       (str): User's question
            timeout (int | None): Timeout seconds
        Returns:
            str: Agent's answer
        """
        if not isinstance(user, User) or not isinstance(question, str):
            err_msg: str = f'Invalid arguments: user: {user}, question: {question}'
            raise TypeError(err_msg)
        try:
            if timeout is None:
                return await self._ask(user=user, question=question)
            traced = check_timeout(self.__class__._ask, timeout=timeout)  # noqa: SLF001
            return await traced(self=self, user=user, question=question)
        except SpecificationError:
            logger.exception('')
            return await self._fallback(user=user, question=question)

    @abstractmethod
    async def _ask(self, user: User, question: str) -> str:
        """Answer user's question
        Args:
            user    (User): User Information
            question (str): User's question
        Returns:
            str: Agent's answer
        """

    @abstractmethod
    async def _fallback(self, user: User, question: str) -> str:
        """Fallback message based on user's question(**Return immidiately**)
        Args:
            user    (User): User Information
            question (str): User's question
        Returns:
            str: Static fallback message
        """


class QAAgentImpl(BaseModel, QAAgent):
    workflow_configs: Annotated[
        dict[str, WorkflowNodeConfig],
        AfterValidator(check_workflow_configs),
        AfterValidator(check_necessary_nodes('answer_node')),
    ] = Field(description='Key equals to WorkflowNodeConfig.node_name')
    fallback_message: str = Field(description='Fallback message is returned when normal flows fail')

    @override
    async def _ask(self, user: User, question: str) -> str:
        initial: QAState = QAState.put_question(question=question)
        response: QAState = await self.workflow.ainvoke(
            initial,
            config={
                'run_name': 'QAAgent.ask',
                'metadata': {
                    'user': user.model_dump(
                        mode='python',
                    )
                },
                'configurable': {'thread_id': user.user_id.user_id},
            },
        )  # type: ignore
        result: str = cast(str, response.answer)
        return result

    @override
    async def _fallback(self, user: User, question: str) -> str:
        log_info = {
            'user': user.model_dump(),
            'question': question,
        }
        log_msg: str = f'fallback: {log_info}'
        logger.warning(log_msg)
        return self.fallback_message

    @cached_property
    def workflow(self) -> Workflow[QAState]:
        answer_node_config: WorkflowNodeConfig = self.workflow_configs['answer_node']
        answer_node_llm: BaseChatModel | None = None
        if answer_node_config.llm_config is not None:
            answer_node_llm = self.__build_llm(llm_config=answer_node_config.llm_config)
        return get_question_answer_workflow(
            answer_node_llm=answer_node_llm, answer_node_template_name=answer_node_config.template_name
        )

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
