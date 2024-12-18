from __future__ import annotations

import os
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Annotated, AsyncGenerator, Callable, Coroutine, Literal

import yaml
from langchain.prompts import BasePromptTemplate, ChatPromptTemplate, PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from pydantic import AfterValidator, BaseModel, Field, PrivateAttr
from typing_extensions import override
from yaml.parser import ParserError

from llm_chatbot_for_messengers.domain.error import ResourceError
from llm_chatbot_for_messengers.domain.repository import MemoryManager, MemoryType


class VolatileMemoryManager(MemoryManager):
    """Store in memory"""

    def __init__(self):
        self._memory_lifecycle: Callable[[], Coroutine[None, None, MemoryType | None]] = (
            self._memory_lifecycle_without_generator()
        )

    @override
    async def acquire_memory(self) -> MemoryType:
        memory: MemoryType | None = await self._memory_lifecycle()
        if memory is None:
            err_msg: str = 'Memory cannot be acquired'
            raise ResourceError(err_msg)
        return memory

    @override
    async def release_memory(self) -> None:
        if (memory := await self._memory_lifecycle()) is not None:
            err_msg: str = f'Memory cannot be released: {memory}'
            raise ResourceError(err_msg)

    def _memory_lifecycle_without_generator(self) -> Callable[[], Coroutine[None, None, MemoryType | None]]:
        memory_generator: AsyncGenerator[MemoryType | None, None] = self._gen_memory()

        async def inner() -> MemoryType | None:
            memory: MemoryType | None = await anext(memory_generator, None)
            return memory

        return inner

    async def _gen_memory(self) -> AsyncGenerator[MemoryType | None, None]:  # noqa: PLR6301
        """To clean memory, use context and generator."""
        # Acquire memory
        yield MemorySaver()


class PersistentMemoryManager(MemoryManager, BaseModel):
    """Store persistantly"""

    conn_uri: Annotated[str, AfterValidator(lambda uri: uri.startswith('postgresql://') and uri)] = Field(
        description='Postgresql connection uri', examples=['postgresql://{username}:{password}@{host}:{port}/{dbname}']
    )
    pool_size: int = Field(description='Connection pool size', gt=0, default=20)
    _memory_lifecycle: Callable[[], Coroutine[None, None, MemoryType | None]] = PrivateAttr()

    def model_post_init(self, __context):
        self._memory_lifecycle = self._memory_lifecycle_without_generator()

    @override
    async def acquire_memory(self) -> MemoryType:
        if (memory := await self._memory_lifecycle()) is None:
            err_msg: str = 'Memory cannot be acquired'
            raise ResourceError(err_msg)
        return memory

    @override
    async def release_memory(self) -> None:
        if (memory := await self._memory_lifecycle()) is not None:
            err_msg: str = f'Memory cannot be released: {memory}'
            raise ResourceError(err_msg)

    def _memory_lifecycle_without_generator(self) -> Callable[[], Coroutine[None, None, MemoryType | None]]:
        memory_generator: AsyncGenerator[MemoryType | None, None] = self._gen_memory()

        async def inner() -> MemoryType | None:
            memory: MemoryType | None = await anext(memory_generator, None)
            return memory

        # Release memory

        return inner

    async def _gen_memory(self) -> AsyncGenerator[MemoryType | None, None]:
        # Acquire memory
        async with AsyncConnectionPool(
            conninfo=self.conn_uri,
            max_size=self.pool_size,
            kwargs={
                'autocommit': True,
                'prepare_threshold': 0,
            },
        ) as pool:
            checkpointer = AsyncPostgresSaver(pool)  # type: ignore
            await checkpointer.setup()
            yield checkpointer


class PromptTemplateParser(ABC):
    @abstractmethod
    def parse(self, raw_template: str) -> ChatPromptTemplate:
        """Parse raw_template to ChatPromptTemplate
        Args:
            raw_template (str) : Prompt template string
        Returns:
            ChatPromptTemplate : Parsed prompt template
        """

    def parse_file(self, node_name: str, template_name: str) -> ChatPromptTemplate:
        if not isinstance(node_name, str) or not isinstance(template_name, str):
            err_msg: str = 'node_name and template_name should be str'
            raise TypeError(err_msg)

        prompt_dir: str | None = os.getenv('PROMPT_DIR')
        if prompt_dir is None:
            prompt_dir_path: Path = Path('src/resources/prompt')
        else:
            try:
                prompt_dir_path = Path(prompt_dir)
            except TypeError as e:
                err_msg = f'PROMPT_DIR({prompt_dir}) environment should indicate valid Path'
                raise ValueError(err_msg) from e

        prompt_path = prompt_dir_path / node_name / template_name
        if not prompt_path.exists():
            err_msg = f"{prompt_path} doesn't exist"
            raise ValueError(err_msg)

        with open(prompt_path, encoding='UTF-8') as fd:
            raw_template: str = fd.read()
            return self.parse(raw_template=raw_template)


class YamlPromptTemplateParser(PromptTemplateParser):
    @override
    def parse(self, raw_template: str) -> ChatPromptTemplate:
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

    @override
    def parse_file(self, node_name: str, template_name: str) -> ChatPromptTemplate:
        return super().parse_file(node_name, f'{template_name}.yaml')


@lru_cache(maxsize=1)
def get_parser() -> PromptTemplateParser:
    return YamlPromptTemplateParser()


def _get_template_of_answer_node(template_name: str | None) -> BasePromptTemplate:
    """Get answer node's Prompt Template
    Args:
        template_name (str | None): Template Name

    Returns:
        BasePromptTemplate: Prompt Template
    """
    match template_name:
        case str():
            return get_parser().parse_file(node_name='answer_node', template_name=template_name)
        case None:
            return ChatPromptTemplate.from_messages([
                ('system', 'Please act as a helpful question-answering agent.'),
                ('human', 'my question is {question}'),
            ])
        case _:
            err_msg: str = f'template_name({template_name}) should be str'
            raise TypeError(err_msg)


def _get_template_of_summary_node(template_name: str | None) -> BasePromptTemplate:
    """Get summary node's Prompt Template
    Args:
        template_name (str | None): Template Name

    Returns:
        BasePromptTemplate: Prompt Template
    """
    match template_name:
        case str():
            return get_parser().parse_file(node_name='summary_node', template_name=template_name)
        case None:
            return ChatPromptTemplate.from_messages([
                ('system', 'Please summrize the document to read quickly main contents.'),
                ('human', '{document}'),
            ])
        case _:
            err_msg: str = f'template_name({template_name}) should be str'
            raise TypeError(err_msg)


def get_template(node_name: str, template_name: str | None = None) -> BasePromptTemplate:
    """Get Prompt Template

    Args:
        node_name (str): Node to use template
        template_name (str | None): Detail template name. Defaults to None.
    Returns:
        BasePromptTemplate: Prompt Template
    Raises:
        RuntimeError: Not Found
    """
    match node_name:
        case 'answer_node':
            return _get_template_of_answer_node(template_name=template_name)
        case 'summary_node':
            return _get_template_of_summary_node(template_name=template_name)
        case _:
            error_msg: str = f'There are no templates for {node_name}'
            raise ValueError(error_msg)

        # Release memory
