from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, Field, PrivateAttr
from typing_extensions import override

from llm_chatbot_for_messengers.domain.chatbot import Chatbot, QAState, QAWithWebSummaryWorkflow, logger
from llm_chatbot_for_messengers.domain.messenger import User
from llm_chatbot_for_messengers.domain.specification import AgentConfig, ChatbotSpecification, TracingSpecification
from llm_chatbot_for_messengers.domain.tracing import Tracing

if TYPE_CHECKING:
    from llm_chatbot_for_messengers.domain.chatbot import MemoryType


class ChatbotFactory:
    pass


class ChatbotImpl(BaseModel, Chatbot):
    config: AgentConfig = Field(description='Agent configuration')
    __workflow: QAWithWebSummaryWorkflow = PrivateAttr(default=None)  # type: ignore

    @override
    async def initialize(self) -> None:
        if self.config.global_configs.memory_manager is not None:
            memory: MemoryType | None = await self.config.global_configs.memory_manager.acquire_memory()
        else:
            memory = None
        self.__workflow = QAWithWebSummaryWorkflow.get_instance(config=self.config.node_configs, memory=memory)

    @override
    async def shutdown(self) -> None:
        if self.config.global_configs.memory_manager is not None:
            await self.config.global_configs.memory_manager.release_memory()

    @override
    async def _ask(self, user: User, question: str) -> str:
        initial: QAState = QAState.put_question(question=question)
        response: QAState = await self.__workflow.ainvoke(
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
        return self.config.global_configs.fallback_message


class ChatbotFactory:
    async def create_chatbot(self, spec: ChatbotSpecification) -> Chatbot:
        """Create a new chatbot
        Args:
            spec (ChatbotSpecification): Chatbot Specification
        Returns:
            Chatbot: Initialized chatbot
        """
        # TODO: impl
        raise NotImplementedError


class TracingFactory:
    async def create_tracing(self, spec: TracingSpecification) -> Tracing:
        """Create a new tracing
        Args:
            spec (TracingSpecification): Tracing Specification
        Returns:
            Tracing: Initialized tracing
        """
        # TODO: impl
        raise NotImplementedError
