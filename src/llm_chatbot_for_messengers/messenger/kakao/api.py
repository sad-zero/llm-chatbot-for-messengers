import logging

from fastapi import FastAPI

from llm_chatbot_for_messengers.messenger.kakao.vo import ChatRequest, ChatResponse, ChatTemplate, SimpleTextOutput

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

app = FastAPI(title='Kakao LLM Chatbot')


@app.post('/kakao/v1/chat')
def chat(body: ChatRequest) -> ChatResponse:
    log_msg: str = f'Request: {body}'
    logger.debug(log_msg)
    # TODO: Manage User ID
    user_id = body.userRequest.user.id
    user_type = body.userRequest.user.type

    # TODO: Manager History
    utterance: str = body.userRequest.utterance

    # TODO: Response
    msg = f"""
    Hello, {user_id}@{user_type}.
    You ask {utterance}.
    """.strip()
    simple_output = SimpleTextOutput.from_text(msg)
    return ChatResponse(template=ChatTemplate.from_outputs(simple_output))
