import asyncio
import os

import pytest
from llm_chatbot_for_messengers.domain.chatbot import (
    QAWithWebSummaryState,
    QAWithWebSummaryWorkflow,
    WebSummaryState,
    WebSummaryWorkflow,
)
from llm_chatbot_for_messengers.domain.configuration import LLMConfig, WorkflowNodeConfig


@pytest.mark.skipif(os.getenv('GITHUB_ACTIONS') == 'true', reason='API KEY cannot be used.')
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('url', 'expected'),
    [('https://en.wikipedia.org/wiki/Spider-Man', 'ok'), ('https://invalid-url.xce.fme.aqw', 'error')],
)
async def test_web_summary_workflow(url, expected):
    # given
    config = {
        'summary_node': WorkflowNodeConfig(
            node_name='summary_node',
            template_name='test',
            llm_config=LLMConfig(model='gpt-4o-mini-2024-07-18', max_tokens=200),
        )
    }
    workflow: WebSummaryWorkflow = WebSummaryWorkflow.get_instance(config=config)
    state = WebSummaryState(url=url)
    # when
    result: WebSummaryState = await asyncio.wait_for(workflow.ainvoke(state), timeout=4)
    # then
    match expected:
        case 'ok':
            assert result.summary is not None
        case 'error':
            assert result.error_message is not None


@pytest.mark.skipif(os.getenv('GITHUB_ACTIONS') == 'true', reason='API KEY cannot be used.')
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('question', 'expected'),
    [
        ('https://en.wikipedia.org/wiki/Spider-Man는 무슨 내용이야?', 'ok'),
        ('된장술밥 어떻게 만들어야 돼?', 'ok'),
        ('https://github.com/astral-sh/ruff는 언제 사용해?', 'ok'),
    ],
)
async def test_qa_with_web_summary_workflow(question, expected):
    # given
    config = {
        'summary_node': WorkflowNodeConfig(
            node_name='summary_node',
            template_name='test',
            llm_config=LLMConfig(model='gpt-4o-mini-2024-07-18', temperature=0.4, max_tokens=100),
        ),
        'answer_node': WorkflowNodeConfig(
            node_name='answer_node',
            template_name='kakao_v3',
            llm_config=LLMConfig(model='gpt-4o-2024-11-20', temperature=0.52, max_tokens=100),
        ),
    }
    workflow: QAWithWebSummaryWorkflow = QAWithWebSummaryWorkflow.get_instance(config=config)
    state = QAWithWebSummaryState(question=question)
    # when
    result: QAWithWebSummaryState = await asyncio.wait_for(workflow.ainvoke(state), timeout=4)
    # then
    match expected:
        case 'ok':
            assert result.answer is not None
        case 'error':
            pytest.fail('Error')
