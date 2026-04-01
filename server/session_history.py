"""
Session History: server-side conversation history per session.

Inspired by Claude Code's context management system. Tracks conversation
turns per session so PRISM can maintain context across multiple messages
and apply context compaction when history gets too long.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Dict, List, Optional

import config

logger = logging.getLogger(__name__)

# Max turns to keep in memory per session (prevents unbounded growth)
_MAX_HISTORY_PER_SESSION = 200
# Max inactive sessions to keep (LRU eviction)
_MAX_SESSIONS = 50


class SessionHistory:
    """
    In-memory per-session conversation history manager.

    Stores conversation turns and provides token counting
    and compaction support.
    """

    def __init__(self) -> None:
        self._histories: Dict[str, List[dict]] = defaultdict(list)
        self._lock = asyncio.Lock()

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        """Append a message to the session history."""
        history = self._histories[session_id]
        history.append({
            "role": role,
            "content": content,
            "compacted": False,
        })

        # Cap history length
        if len(history) > _MAX_HISTORY_PER_SESSION:
            self._histories[session_id] = history[-_MAX_HISTORY_PER_SESSION:]

        # LRU eviction of inactive sessions
        if len(self._histories) > _MAX_SESSIONS:
            oldest = list(self._histories.keys())[0]
            if oldest != session_id:
                del self._histories[oldest]

    def get_history(self, session_id: str) -> List[dict]:
        """Return the current conversation history for a session."""
        return list(self._histories.get(session_id, []))

    def get_inference_history(self, session_id: str) -> List[dict]:
        """
        Return history formatted for inference (role + content only,
        excluding the most recent user message which is passed separately).
        """
        history = self._histories.get(session_id, [])
        if not history:
            return []
        # Exclude the last message (it's the current user message)
        prior = history[:-1] if history else []
        return [{"role": h["role"], "content": h["content"]} for h in prior]

    def get_token_estimate(self, session_id: str) -> int:
        """
        Approximate token count for a session's history.
        Uses a simple heuristic: 1 token per 4 characters.
        """
        history = self._histories.get(session_id, [])
        total_chars = sum(len(h["content"]) for h in history)
        return total_chars // 4

    def needs_compaction(self, session_id: str) -> bool:
        """Check if session history exceeds the compaction threshold."""
        return self.get_token_estimate(session_id) > config.COMPACTION_TOKEN_THRESHOLD

    async def compact(self, session_id: str, summary: str) -> None:
        """
        Replace older messages with a summary, keeping recent turns intact.
        """
        async with self._lock:
            history = self._histories.get(session_id, [])
            keep_count = config.COMPACTION_KEEP_RECENT

            if len(history) <= keep_count:
                return

            # Keep only recent turns, prepend summary
            recent = history[-keep_count:]
            compacted_entry = {
                "role": "system",
                "content": f"[Previous conversation summary: {summary}]",
                "compacted": True,
            }
            self._histories[session_id] = [compacted_entry] + recent

        logger.info(
            "Compacted session %s: %d turns summarized, %d kept.",
            session_id, len(history) - keep_count, keep_count,
        )

    def reset(self, session_id: str) -> None:
        """Clear all history for a session."""
        self._histories.pop(session_id, None)

    def get_session_count(self) -> int:
        """Return the number of active sessions with history."""
        return len(self._histories)


# Module-level singleton
_history_instance: Optional[SessionHistory] = None


def get_session_history() -> SessionHistory:
    """Return (or lazily create) the global SessionHistory."""
    global _history_instance
    if _history_instance is None:
        _history_instance = SessionHistory()
    return _history_instance
