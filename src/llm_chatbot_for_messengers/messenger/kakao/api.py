import logging
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, FastAPI, Request, Response

from llm_chatbot_for_messengers.core.entity.agent import QAAgent
from llm_chatbot_for_messengers.messenger.kakao.container import get_qa_agent, get_rate_limit_strategy, manage_resources
from llm_chatbot_for_messengers.messenger.kakao.vo import ChatRequest, ChatResponse, ChatTemplate, SimpleTextOutput
from llm_chatbot_for_messengers.messenger.vo import MessengerRequest

if TYPE_CHECKING:
    from llm_chatbot_for_messengers.messenger.middleware.rate_limit import RateLimitStrategy

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
app = FastAPI(title='Kakao LLM Chatbot', lifespan=manage_resources)


@app.post('/kakao/v1/chat', response_model=None)
async def chat(body: ChatRequest, qa_agent: Annotated[QAAgent, Depends(get_qa_agent)]) -> ChatResponse:
    user = body.userRequest.user.to()
    answer = await qa_agent.ask(user=user, question=body.userRequest.utterance, timeout=4)
    simple_output = SimpleTextOutput.from_text(answer)
    return ChatResponse(template=ChatTemplate.from_outputs(simple_output))


@app.middleware('http')
async def add_rate_limit(request: Request, call_next) -> Response:
    rate_limit_strategy: RateLimitStrategy = get_rate_limit_strategy()
    match request.url.path:
        case '/kakao/v1/chat':
            body: ChatRequest = ChatRequest.model_validate(await request.json())
            messenger_request: MessengerRequest = MessengerRequest(user=body.userRequest.user.to(), messenger='kakao')
            is_accepted, next_available_sec = await rate_limit_strategy.accept(request=messenger_request)
            if is_accepted is True:
                return await call_next(request)
        case _:
            return await call_next(request)
    # Rate over
    return Response(
        status_code=429,
        headers={
            'Content-Type': 'application/json',
            'Retry-After': f'{next_available_sec}',
        },
    )
