from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from llm_chatbot_for_messengers.core.entity.user import User  # noqa: TCH001


class MessengerRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    user: User = Field(description='User information')
    messenger: str = Field(description='Messenger Id')
