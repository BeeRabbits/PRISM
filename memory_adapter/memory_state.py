"""
Per-session Titans memory state manager.

Each chat session gets its own memory tensor that evolves
over the course of the session. Memory resets between
unrelated sessions but persists within a session
across turns.

Thread-safe: uses a dict protected by asyncio.Lock.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Optional

import torch

import config

logger = logging.getLogger(__name__)


class MemoryStateManager:
    """
    Manages per-session Titans memory tensors.

    Sessions are identified by session_id strings.
    Memory is stored as float32 tensors on CPU to avoid
    wasting GPU memory for idle sessions; it is moved to
    the active device only during inference.
    """

    def __init__(
        self,
        memory_size: int = 512,
        memory_dim: int = 512,
    ) -> None:
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self._store: Dict[str, torch.Tensor] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get(self, session_id: str) -> Optional[torch.Tensor]:
        """
        Return the current memory tensor for session_id, or None
        if no memory has been initialised for this session yet.
        """
        async with self._lock:
            return self._store.get(session_id)

    async def get_or_init(self, session_id: str) -> torch.Tensor:
        """
        Return the memory for session_id, initialising it to zeros
        if it doesn't exist yet.
        """
        async with self._lock:
            if session_id not in self._store:
                logger.debug("Initialising memory for session %s", session_id)
                self._store[session_id] = torch.zeros(
                    self.memory_size, self.memory_dim, dtype=torch.float32
                )
            return self._store[session_id]

    async def update(self, session_id: str, new_memory: torch.Tensor) -> None:
        """
        Update the stored memory for session_id.

        Args:
            session_id: Session identifier.
            new_memory: [memory_size, memory_dim] tensor (any device).
                        Will be detached and moved to CPU for storage.
        """
        stored = new_memory.detach().cpu().float()
        async with self._lock:
            self._store[session_id] = stored

    async def reset(self, session_id: str) -> None:
        """Clear the memory for session_id (e.g. when starting a new topic)."""
        async with self._lock:
            if session_id in self._store:
                del self._store[session_id]
                logger.info("Memory reset for session %s", session_id)

    async def reset_all(self) -> None:
        """Clear all session memories (e.g. on server restart)."""
        async with self._lock:
            self._store.clear()
            logger.info("All session memories cleared.")

    def active_sessions(self) -> list[str]:
        """Return list of session IDs with active memory."""
        return list(self._store.keys())

    def session_count(self) -> int:
        return len(self._store)


# Module-level singleton, shared across the FastAPI app
_manager: Optional[MemoryStateManager] = None


def get_memory_manager() -> MemoryStateManager:
    """Return (or lazily create) the global MemoryStateManager."""
    global _manager
    if _manager is None:
        _manager = MemoryStateManager(
            memory_size=config.TITANS_MEMORY_SIZE,
            memory_dim=config.TITANS_MEMORY_DIM,
        )
    return _manager
