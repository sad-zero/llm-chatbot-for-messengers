from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from llm_chatbot_for_messengers.core.vo import UserId


class User(BaseModel):
    user_id: UserId = Field(description="User's Unique Id")
    user_name: str | None = Field(description="User's name", default=None)
