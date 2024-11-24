from __future__ import annotations

from pydantic import BaseModel, Field


class User(BaseModel):
    id: str = Field(description="User's id")
    type: str = Field(description="User's type")


class UserRequest(BaseModel):
    utterance: str = Field(description='Current utterance')
    user: User = Field(description="User's detail")


class ChatRequest(BaseModel):
    userRequest: UserRequest = Field(description="User's information")  # noqa: N815


class _SimpleText(BaseModel):
    text: str = Field(description='Text')


class SimpleTextOutput(BaseModel):
    simpleText: _SimpleText = Field(description='Text response')  # noqa: N815

    @classmethod
    def from_text(cls, text: str) -> SimpleTextOutput:
        return cls(simpleText=_SimpleText(text=text))


class ChatTemplate(BaseModel):
    outputs: list[SimpleTextOutput] = Field(description='Chat detail')

    @classmethod
    def from_outputs(cls, *outputs: SimpleTextOutput) -> ChatTemplate:
        return cls(outputs=list(outputs))


class ChatResponse(BaseModel):
    version: str = Field(description='Kakao IF version', default='2.0')
    template: ChatTemplate = Field(description='Response detail')
