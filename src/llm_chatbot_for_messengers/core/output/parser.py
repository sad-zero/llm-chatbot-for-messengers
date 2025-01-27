import os
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
)
from typing_extensions import override
from yaml.parser import ParserError


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
