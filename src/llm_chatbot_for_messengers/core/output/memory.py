from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeAlias

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import override

MemoryType: TypeAlias = BaseCheckpointSaver


class MemoryManager(ABC):
    @abstractmethod
    def get_memory(self) -> MemoryType:
        """Return a langgraph checkpoint."""


class VolatileMemoryManager(MemoryManager):
    """Store in memory"""

    def __init__(self):
        self.__memory = MemorySaver()

    @override
    def get_memory(self) -> MemoryType:
        return self.__memory
