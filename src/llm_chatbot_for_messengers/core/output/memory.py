from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Callable, Coroutine, TypeAlias

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import override

from llm_chatbot_for_messengers.core.error import ResourceError

MemoryType: TypeAlias = BaseCheckpointSaver


class MemoryManager(ABC):
    @abstractmethod
    async def acquire_memory(self) -> MemoryType:
        pass

    @abstractmethod
    async def release_memory(self) -> None:
        pass


class VolatileMemoryManager(MemoryManager):
    """Store in memory"""

    def __init__(self):
        self._memory_lifecycle: Callable[[], Coroutine[None, None, MemoryType | None]] = (
            self._memory_lifecycle_without_generator()
        )

    @override
    async def acquire_memory(self) -> MemoryType:
        memory: MemoryType | None = await self._memory_lifecycle()
        if memory is None:
            err_msg = 'Memory cannot be acquired'
            raise ResourceError(err_msg)
        return memory

    @override
    async def release_memory(self) -> None:
        if (memory := await self._memory_lifecycle()) is not None:
            err_msg = f'Memory cannot be released: {memory}'
            raise ResourceError(err_msg)

    def _memory_lifecycle_without_generator(self) -> Callable[[], Coroutine[None, None, MemoryType | None]]:
        memory_generator: AsyncGenerator[MemoryType | None, None] = self._gen_memory()

        async def inner(self=self) -> MemoryType | None:  # noqa: ARG001
            memory: MemoryType | None = await anext(memory_generator, None)
            return memory

        return inner

    async def _gen_memory(self) -> AsyncGenerator[MemoryType | None, None]:  # noqa: PLR6301
        """To clean memory, use context and generator."""
        # Acquire memory
        yield MemorySaver()
        # Release memory
