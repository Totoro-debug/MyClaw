"""Compatibility re-exports for the renamed Dream boundary."""

from myclaw.memory.dream import (
    Dream,
    DreamEditFileTool,
    DreamModelRouter,
    DreamReadFileTool,
    DreamResult,
)
from myclaw.memory.manager import MemoryManager, MemoryPathDeniedError
from myclaw.memory.store import (
    MemoryStore,
    WorkspaceFileMemoryStore,
    WorkspaceJsonlSummaryStore,
)

MemoryEditFileTool = DreamEditFileTool
MemoryReadFileTool = DreamReadFileTool
MemoryTaskModelRouter = DreamModelRouter
MemoryTaskResult = DreamResult


__all__ = [
    "Dream",
    "MemoryEditFileTool",
    "MemoryManager",
    "MemoryPathDeniedError",
    "MemoryReadFileTool",
    "MemoryStore",
    "MemoryTaskModelRouter",
    "MemoryTaskResult",
    "WorkspaceFileMemoryStore",
    "WorkspaceJsonlSummaryStore",
]
