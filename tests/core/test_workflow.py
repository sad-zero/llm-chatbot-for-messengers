import asyncio
import os

import pytest
from llm_chatbot_for_messengers.core.configuration import LLMConfig, WorkflowNodeConfig
from llm_chatbot_for_messengers.core.workflow.qa import WebSummaryWorkflow
from llm_chatbot_for_messengers.core.workflow.vo import WebSummaryState


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
    state = WebSummaryState(url=url)  # type: ignore
    # when
    result: WebSummaryState = await asyncio.wait_for(workflow.ainvoke(state), timeout=4)
    # then
    match expected:
        case 'ok':
            assert result.summary is not None
        case 'error':
            assert result.error_message is not None
