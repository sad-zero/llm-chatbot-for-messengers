import asyncio

import pytest
from llm_chatbot_for_messengers.core.vo import LLMConfig, WebSummaryState, WorkflowNodeConfig
from llm_chatbot_for_messengers.core.workflow.qa import WebSummaryWorkflow


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
    state = WebSummaryState(url=url)  # type: ignore
    # when
    async with asyncio.timeout(4):
        result: WebSummaryState = await workflow.ainvoke(state)
    # then
    match expected:
        case 'ok':
            assert result.summary is not None
        case 'error':
            assert result.error_message is not None
