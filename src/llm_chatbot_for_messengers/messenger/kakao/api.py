import logging
from typing import Annotated

from fastapi import Depends, FastAPI

from llm_chatbot_for_messengers.core.agent import QAAgent
from llm_chatbot_for_messengers.messenger.kakao.container import get_qa_agent
from llm_chatbot_for_messengers.messenger.kakao.vo import ChatRequest, ChatResponse, ChatTemplate, SimpleTextOutput

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)
app = FastAPI(title='Kakao LLM Chatbot')


@app.post('/kakao/v1/chat')
async def chat(body: ChatRequest, qa_agent: Annotated[QAAgent, Depends(get_qa_agent)]) -> ChatResponse:
    user = body.userRequest.user.to()
    answer = await qa_agent.ask(user=user, question=body.userRequest.utterance)
    simple_output = SimpleTextOutput.from_text(answer)
    return ChatResponse(template=ChatTemplate.from_outputs(simple_output))
