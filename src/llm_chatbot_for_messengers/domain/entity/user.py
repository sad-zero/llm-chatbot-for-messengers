from __future__ import annotations

from pydantic import BaseModel, Field

from llm_chatbot_for_messengers.domain.vo import MessengerId, UserId  # noqa: TCH001


class User(BaseModel):
    messenger_id: MessengerId | None = Field(description="Messenger's Id", default=None)
    user_id: UserId = Field(description="User's Unique Id")
    user_name: str | None = Field(description="User's name", default=None)
