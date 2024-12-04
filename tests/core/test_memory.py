from llm_chatbot_for_messengers.core.output.memory import MemoryManager, MemoryType, VolatileMemoryManager


def test_volatile_memory():
    # given
    manager: MemoryManager = VolatileMemoryManager()
    # when
    memory: MemoryType = manager.get_memory()
    # then
    assert isinstance(manager, MemoryManager)
    assert isinstance(memory, MemoryType)
