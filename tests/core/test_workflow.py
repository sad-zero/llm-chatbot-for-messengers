from typing import TYPE_CHECKING

import pytest
from langchain_community.llms.fake import FakeListLLM
from llm_chatbot_for_messengers.core.workflow import get_question_answer_workflow

if TYPE_CHECKING:
    from llm_chatbot_for_messengers.core.vo import QAState


@pytest.mark.asyncio(loop_scope='function')
async def test_get_question_answer_workflow():
    # given
    answer_node_llm = FakeListLLM(responses=['Hi! What can I do for you?'])
    workflow = get_question_answer_workflow(answer_node_llm=answer_node_llm)  # type: ignore
    question_state: QAState = {
        'question': 'Hello!',
    }
    # when
    answer_state: QAState = await workflow.ainvoke(question_state, debug=True)  # type: ignore
    # then
    assert answer_state['question'] == question_state['question']
    assert 'answer' in answer_state
    assert answer_state['answer'] == 'Hi! What can I do for you?'
