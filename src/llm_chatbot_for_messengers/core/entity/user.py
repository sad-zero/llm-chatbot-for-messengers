from __future__ import annotations

from pydantic import BaseModel, Field

from llm_chatbot_for_messengers.core.vo import UserId  # noqa: TCH001


class User(BaseModel):
    user_id: UserId = Field(description="User's Unique Id")
    user_name: str | None = Field(description="User's name", default=None)
