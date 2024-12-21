from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from typing import AsyncGenerator, Generator, Literal
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel
import yaml
from yaml.parser import ParserError
from llm_chatbot_for_messengers.domain.chatbot import Chatbot, Memory, Prompt, Workflow
from llm_chatbot_for_messengers.domain.error import FactoryError
from llm_chatbot_for_messengers.domain.repository import ChatbotRepository
from llm_chatbot_for_messengers.domain.specification import ChatbotSpecification, MemorySpecification, PromptSpecification, TracingSpecification, WorkflowSpecification
from llm_chatbot_for_messengers.domain.tracing import Tracing

logger = logging.getLogger(__name__)
# class ChatbotImpl(BaseModel, Chatbot):
#     config: AgentConfig = Field(description='Agent configuration')
#     __workflow: QAWithWebSummaryWorkflow = PrivateAttr(default=None)  # type: ignore

#     @override
#     async def initialize(self) -> None:
#         if self.config.global_configs.memory_manager is not None:
#             memory: MemoryType | None = await self.config.global_configs.memory_manager.acquire_memory()
#         else:
#             memory = None
#         self.__workflow = QAWithWebSummaryWorkflow.get_instance(config=self.config.node_configs, memory=memory)

#     @override
#     async def shutdown(self) -> None:
#         if self.config.global_configs.memory_manager is not None:
#             await self.config.global_configs.memory_manager.release_memory()

#     @override
#     async def _ask(self, user: User, question: str) -> str:
#         initial: QAState = QAState.put_question(question=question)
#         response: QAState = await self.__workflow.ainvoke(
#             initial,
#             config={
#                 'run_name': 'QAAgent.ask',
#                 'metadata': {
#                     'user': user.model_dump(
#                         mode='python',
#                     )
#                 },
#                 'configurable': {'thread_id': user.user_id.user_id},
#             },
#         )  # type: ignore
#         result: str = cast(str, response.answer)
#         return result

#     @override
#     async def _fallback(self, user: User, question: str) -> str:
#         log_info = {
#             'user': user.model_dump(),
#             'question': question,
#         }
#         log_msg: str = f'fallback: {log_info}'
#         logger.warning(log_msg)
#         return self.config.global_configs.fallback_message


class ChatbotFactory:
    @asynccontextmanager
    async def create_chatbot(self, spec: ChatbotSpecification) -> AsyncGenerator[Chatbot, None]:
        """Create a new chatbot
        Args:
            spec (ChatbotSpecification): Chatbot Specification
        Returns:
            Chatbot: Initialized chatbot
        """
        try:
            prompts: list[Prompt] = await self.__create_prompts(spec.prompt_specs)
            async with self.__create_memory(spec.memory_spec) as memory:
                workflow: Workflow = await self.__create_workflow(spec=spec.workflow_spec, memory=memory, prompts=prompts)

                chatbot: Chatbot = Chatbot(
                    workflow=workflow,
                    memory=memory,
                    prompts=prompts,
                    timeout=spec.timeout,
                    fallback_message=spec.fallback_message,
                )
                yield chatbot
        except Exception as e:
            err_msg: str = f"There are some problems when creating chatbot with spec: {spec:r}" 
            raise FactoryError(err_msg) from e

    
    @asynccontextmanager
    async def __create_memory(self, spec: MemorySpecification) -> AsyncGenerator[Memory, None]:
        err_msg: str = f"There are some problems when creating memory with spec: {spec:r}"
        try:
           match spec.type_:
                case "volatile":
                   async with MemorySaver() as saver:
                       yield saver
                case "persistant":
                   async with AsyncConnectionPool(
                       conninfo=spec.conn_uri,
                       max_size=spec.conn_pool_size,
                       kwargs={
                           'autocommit': True,
                           'prepare_threshold': 0,
                       },
                   ) as pool:
                        yield AsyncPostgresSaver(pool)
                case _:
                   raise FactoryError(err_msg)
                
        except Exception as e:
            raise FactoryError(err_msg) from e

    async def __create_prompts(self, specs: list[PromptSpecification]) -> list[Prompt]:
        prompt_dir: str | None = os.getenv('PROMPT_DIR')
        if prompt_dir is None:
            prompt_dir_path: Path = Path('src/resources/prompt')
        else:
            try:
                prompt_dir_path = Path(prompt_dir)
            except TypeError as e:
                err_msg = f'PROMPT_DIR({prompt_dir}) environment should indicate valid Path'
                raise ValueError(err_msg) from e

        result: list[Prompt] = [] 
        for spec in specs:
            prompt_path: Path = prompt_dir_path / spec.node / spec.name
            if not prompt_path.exists():
                err_msg = f"{prompt_path} doesn't exist"
                raise ValueError(err_msg)

            with open(prompt_path, encoding='UTF-8') as fd:
                raw_template: str = fd.read()
                template: ChatPromptTemplate = self.__parse(raw_template)
                prompt = Prompt(node=spec.node, name=spec.name, template=template) 
            result.append(prompt)
        return result 

    def __parse(self, raw_template: Path) -> ChatPromptTemplate:
        try:
            loaded_template: dict = yaml.load(raw_template, Loader=yaml.SafeLoader)
            if not isinstance(loaded_template, dict):
                err_msg: str = 'Invalid template format'
                raise TypeError(err_msg)  # noqa: TRY301
            templates: list[dict] | None = loaded_template.get('templates')
            if templates is None:
                err_msg = "Templates property doesn't exist"
                raise TypeError(err_msg)  # noqa: TRY301
            if not isinstance(templates, list):
                err_msg = 'Templates should be list'
                raise TypeError(err_msg)  # noqa: TRY301

            prompt_templates: list[tuple[Literal['system', 'human', 'ai'], str]] = []
            template_properties: set[str] = {'type', 'input_variables', 'template'}
            for template in templates:
                if not isinstance(template, dict):
                    err_msg = 'Template should be dict'
                    raise TypeError(err_msg)  # noqa: TRY301
                if template_properties.intersection(template.keys()) != template_properties:
                    missing_properties = template_properties - template.keys()
                    err_msg = f'Missing properties: {missing_properties}'
                    raise ValueError(err_msg)  # noqa: TRY301

                input_variables: list[str] = template['input_variables']
                template_content: str = template['template']

                if not isinstance(input_variables, list):
                    err_msg = 'Input variables should be list'
                    raise TypeError(err_msg)  # noqa: TRY301
                if not isinstance(template_content, str):
                    err_msg = 'Template content should be str'
                    raise TypeError(err_msg)  # noqa: TRY301

                PromptTemplate(
                    template=template_content, input_variables=input_variables, validate_template=True
                )  # Validate template
                match type_ := template['type']:
                    case 'system' | 'human' | 'ai':
                        prompt_templates.append((type_, template_content.strip()))
                    case _:
                        err_msg = f'Invalid template type: {type_}'
                        raise ValueError(err_msg)  # noqa: TRY301
            return ChatPromptTemplate.from_messages(prompt_templates)
        except Exception as e:
            err_msg = f'Parsing fails: {raw_template}'
            raise ParserError(err_msg) from e
    
    async def __create_workflow(self, spec: WorkflowSpecification, memory: Memory, prompts: list[Prompt]) -> Workflow:
        pass


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
