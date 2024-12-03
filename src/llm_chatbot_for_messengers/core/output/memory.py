from __future__ import annotations

from contextlib import AbstractAsyncContextManager, AbstractContextManager

from langgraph.checkpoint.base import BaseCheckpointSaver


class AgentMemory(BaseCheckpointSaver[str], AbstractContextManager, AbstractAsyncContextManager):
    pass
