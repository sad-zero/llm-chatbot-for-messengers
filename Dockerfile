FROM ubuntu:latest AS builder

WORKDIR /llm-chatbot-for-messengers
COPY . .

ENV PATH=$PATH:/root/.local/bin
RUN apt-get update
RUN apt-get install pipx -y
RUN pipx install hatch
RUN hatch config set dirs.env.virtual .venv
RUN hatch run builder:build-wheel

FROM python:3.11 AS runner
WORKDIR /llm-chatbot-for-messengers
COPY --from=builder /llm-chatbot-for-messengers/dist/llm_chatbot_for_messengers*.whl .
COPY --from=builder /llm-chatbot-for-messengers/src/resources src/resources
RUN pip install llm_chatbot_for_messengers*.whl

ENV OPENAI_API_KEY "<<YOUR OPENAI API KEY>>"
# If you want to trace agent by LANGSMITH, change to true.
ENV LANGCHAIN_TRACING_V2 "false"
ENV LANGCHAIN_API_KEY "<<YOUR LANGSMITH API KEY>>"
ENV LANGCHAIN_PROJECT "<<YOUR LANGSMITH PROJECT NAME>>"
EXPOSE 8000
ENTRYPOINT [ "kakao_chatbot" ]
