import json

import pytest
from langchain_community.llms.fake import FakeListLLM
from llm_chatbot_for_messengers.core.vo import QAState
from llm_chatbot_for_messengers.core.workflow import get_question_answer_workflow


@pytest.mark.asyncio(loop_scope='function')
async def test_get_question_answer_workflow():
    # given
    answer_node_llm = FakeListLLM(responses=[json.dumps({'answer': 'Hi! What can I do for you?'})])
    workflow = get_question_answer_workflow(answer_node_llm=answer_node_llm)  # type: ignore
    question_state: QAState = QAState.put_question(question='Hello!')
    # when
    answer_state: QAState = await workflow.ainvoke(
        question_state,
        debug=True,
        config={
            'configurable': {
                'thread_id': 'test-id',
            }
        },
    )  # type: ignore
    # then
    assert answer_state.question == question_state.question
    assert answer_state.answer == 'Hi! What can I do for you?'
