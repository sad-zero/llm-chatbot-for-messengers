"""Entrypoints"""

import uvicorn


def run_kakao_chatbot():
    uvicorn.run(
        app='llm_chatbot_for_messengers.messenger.kakao.api:app',
        host='0.0.0.0',  # noqa: S104
        port=8000,
        workers=4,
        log_level='info',
        use_colors=False,
    )
