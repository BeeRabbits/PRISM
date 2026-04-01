"""
Compaction Service: summarizes long conversation histories.

Inspired by Claude Code's context compaction algorithm that auto-compacts
at ~95% context capacity, preserving key decisions while reducing tokens.

Uses the local PRISM model for summarization (zero API cost).
"""

from __future__ import annotations

import logging
from typing import Optional

import config

logger = logging.getLogger(__name__)


class CompactionService:
    """
    Summarizes older conversation history to keep context focused.

    Uses the local model for summarization by default (no API cost).
    """

    def __init__(self, engine=None) -> None:
        self._engine = engine

    async def compact_if_needed(self, session_id: str) -> bool:
        """
        Check if a session needs compaction and perform it if so.

        Returns True if compaction was performed.
        """
        from server.session_history import get_session_history

        history_mgr = get_session_history()
        if not history_mgr.needs_compaction(session_id):
            return False

        history = history_mgr.get_history(session_id)
        keep_count = config.COMPACTION_KEEP_RECENT

        if len(history) <= keep_count:
            return False

        # Get the older messages to summarize
        older = history[:-keep_count]
        older_text = "\n".join(
            f"{h['role'].capitalize()}: {h['content'][:200]}"
            for h in older
            if not h.get("compacted")
        )

        if not older_text.strip():
            return False

        # Generate summary using local model
        summary = await self._generate_summary(older_text)
        if summary:
            await history_mgr.compact(session_id, summary)
            return True

        return False

    async def _generate_summary(self, conversation_text: str) -> Optional[str]:
        """
        Generate a summary of conversation history.

        Uses the local model if available, otherwise creates a
        simple extractive summary.
        """
        if self._engine is not None:
            try:
                summary_prompt = (
                    "Summarize the following conversation in 2-3 sentences. "
                    "Preserve key facts, preferences, decisions, and any "
                    "personal details mentioned. Be concise.\n\n"
                    f"{conversation_text[:2000]}"
                )
                response, _ = await self._engine.generate(
                    session_id="__compaction__",
                    message=summary_prompt,
                    system_prompt="You are a summarization assistant. Output only the summary.",
                )
                return response.strip()
            except Exception:
                logger.debug("Local model summarization failed, using extractive fallback.")

        # Extractive fallback: take first and last few messages
        lines = conversation_text.strip().split("\n")
        if len(lines) <= 4:
            return conversation_text
        summary_lines = lines[:2] + ["..."] + lines[-2:]
        return " ".join(summary_lines)


# Module-level singleton
_compaction_instance: Optional[CompactionService] = None


def get_compaction_service() -> CompactionService:
    """Return (or lazily create) the global CompactionService."""
    global _compaction_instance
    if _compaction_instance is None:
        _compaction_instance = CompactionService()
    return _compaction_instance


def set_compaction_service(service: CompactionService) -> None:
    """Set the global CompactionService with engine reference."""
    global _compaction_instance
    _compaction_instance = service
