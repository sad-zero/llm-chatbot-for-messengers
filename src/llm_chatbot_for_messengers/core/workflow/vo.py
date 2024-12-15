from __future__ import annotations

import json
from typing import Annotated, Literal, Self

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages import AnyMessage  # noqa: TCH002
from langgraph.graph import add_messages  # noqa: TCH002
from pydantic import BaseModel, ConfigDict, Field, HttpUrl
from typing_extensions import TypedDict

# Define Workflow States


class SummaryNodeDocument(TypedDict):
    title: str | None
    chunks: list[str]


class WebSummaryState(BaseModel):
    url: HttpUrl = Field(description='Website url')
    html_document: str | None = Field(description='HTML document crawled at url', default=None)
    document: SummaryNodeDocument | None = Field(description='Parsed document', default=None)
    error_message: str | None = Field(description='Error message', default=None)
    summary: str | None = Field(description="Agent's summary", default=None)

    @classmethod
    def initialize(cls, url: str) -> Self:
        return cls(url=url)  # type: ignore


class QAState(BaseModel):
    question: str | None = Field(description="User's question", default=None)
    context: str = Field(description='Context to answer the question', default='No context')
    answer: str | None = Field(description="Agent's answer", default=None)
    messages: Annotated[list[AnyMessage], add_messages] = Field(description='Chat histories', default_factory=list)

    @classmethod
    def put_question(cls, question: str) -> Self:
        return cls(
            question=question,
            messages=[HumanMessage(question)],
        )

    @classmethod
    def put_answer(cls, answer: str) -> Self:
        return cls(
            answer=answer,
            messages=[AIMessage(answer)],
        )

    def get_formatted_messages(self) -> str:
        """Format messages as tuples
        Returns:
            str: [
                ("system", ...),
                ("human", ...),
                ("ai", ...),
            ]
        """
        result: list[tuple[Literal['system', 'human', 'ai'], str | list[str | dict]]] = []
        for message in self.messages:
            if isinstance(message, SystemMessage):
                result.append(('system', message.content))
            elif isinstance(message, HumanMessage):
                result.append(('human', message.content))
            elif isinstance(message, AIMessage):
                result.append(('ai', message.content))
            else:
                err_msg = f'Invalid message type: {message}'
                raise TypeError(err_msg)
        return json.dumps(result, indent=4, ensure_ascii=False)


class QAWithWebSummaryState(BaseModel):
    question: str | None = Field(description="User's question", default=None)
    context: str | None = Field(description='Context to answer the question', default=None)
    answer: str | None = Field(description="Agent's answer", default=None)


# Define LLM Response formats


class AnswerNodeResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    sentences: list[str] = Field(description="Answer node's output")


class SummaryNodeResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    summary: str = Field(description="Summary node's output")
