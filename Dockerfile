FROM ubuntu:latest AS builder

WORKDIR /llm-chatbot-for-messengers
COPY . .

ENV PATH=$PATH:/root/.local/bin
RUN apt-get update
RUN apt-get install pipx -y
RUN pipx install hatch
RUN hatch config set dirs.env.virtual .venv
RUN hatch run builder:build-wheel

FROM python:3.11-slim AS runner
WORKDIR /llm-chatbot-for-messengers
COPY --from=builder /llm-chatbot-for-messengers/dist/llm_chatbot_for_messengers*.whl .
RUN pip install llm_chatbot_for_messengers*.whl

EXPOSE 8000
ENTRYPOINT [ "kakao_chatbot" ]
