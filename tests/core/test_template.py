import re
from typing import Literal

import pytest
from langchain.prompts import BasePromptTemplate
from llm_chatbot_for_messengers.core.template import get_template


@pytest.mark.parametrize(
    ('node_name', 'template_name', 'expected'),
    [
        ('answer_node', None, 'ok'),
        ('answer_node', 'invalid', 'error'),
        ('invalid', None, 'error'),
    ],
)
def test_get_template(node_name: str, template_name: str, expected: Literal['ok', 'error']):
    # when
    if expected == 'ok':
        assert isinstance(get_template(node_name=node_name, template_name=template_name), BasePromptTemplate)
    else:
        with pytest.raises(ValueError, match=re.compile(r'There are|exist')):
            get_template(node_name=node_name, template_name=template_name)
