import os

import pytest
from llm_chatbot_for_messengers.core.output.memory import (
    MemoryManager,
    MemoryType,
    PersistantMemoryManager,
    VolatileMemoryManager,
)


@pytest.mark.asyncio
async def test_volatile_memory():
    # given
    manager: MemoryManager = VolatileMemoryManager()
    # when
    memory: MemoryType = await manager.acquire_memory()
    # then
    assert isinstance(memory, MemoryType)
    await manager.release_memory()


@pytest.mark.asyncio
async def test_persistant_memory():
    # given
    manager: MemoryManager = PersistantMemoryManager(conn_uri=os.getenv('CORE_DB_URI'))  # type: ignore
    # when
    memory: MemoryType = await manager.acquire_memory()
    # then
    assert isinstance(memory, MemoryType)
    await manager.release_memory()
