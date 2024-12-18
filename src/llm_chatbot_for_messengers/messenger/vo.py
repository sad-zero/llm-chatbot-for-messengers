from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from llm_chatbot_for_messengers.domain.entity.messenger import Messenger  # noqa: TCH001
from llm_chatbot_for_messengers.domain.entity.user import User  # noqa: TCH001


class MessengerRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    user: User = Field(description="User's information")
    messenger: Messenger = Field(description="Messenger's information")
