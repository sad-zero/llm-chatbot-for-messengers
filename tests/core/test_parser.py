import pytest
from langchain.prompts import ChatPromptTemplate
from llm_chatbot_for_messengers.core.output.parser import YamlPromptTemplateParser


def test_yaml_parser():
    # given
    parser = YamlPromptTemplateParser()
    node_name = 'test_node'
    template_name = 'test'
    expected = ChatPromptTemplate.from_messages([
        ('system', 'Please act as a helpful question-answering agent.'),
        ('human', 'my question is {question}'),
    ])
    # when
    actual = parser.parse_file(node_name=node_name, template_name=template_name)
    # then
    assert expected == actual


def test_fail():
    pytest.fail(reason='test')
