import pytest
from llm_chatbot_for_messengers.core.output.memory import MemoryManager, MemoryType, VolatileMemoryManager


@pytest.mark.asyncio
async def test_volatile_memory():
    # given
    manager: MemoryManager = VolatileMemoryManager()
    # when
    memory: MemoryType = await manager.acquire_memory()
    # then
    assert isinstance(memory, MemoryType)
    await manager.release_memory()
