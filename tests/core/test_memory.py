import os

import pytest
from llm_chatbot_for_messengers.core.output.memory import (
    MemoryManager,
    MemoryType,
    PersistentMemoryManager,
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


@pytest.mark.skipif(os.getenv('GITHUB_ACTIONS') == 'true', reason='Database cannot be established in git action.')
@pytest.mark.asyncio
async def test_persistent_memory():
    # given
    manager: MemoryManager = PersistentMemoryManager(conn_uri=os.getenv('CORE_DB_URI'))  # type: ignore
    # when
    memory: MemoryType = await manager.acquire_memory()
    # then
    assert isinstance(memory, MemoryType)
    await manager.release_memory()
