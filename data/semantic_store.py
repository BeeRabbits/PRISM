"""
SemanticStore: async store for validated semantic memories.

Semantic memories are distilled from clusters of episodic memories by
DreamConsolidation. They go through a provisional → confirmed promotion
cycle before being allowed into LoRA training data.

Schema is defined in data/experience_log.py (SemanticMemoryRow) and
created by ExperienceLogger.init().
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import config
from data.experience_log import Base, SemanticMemoryRow

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are PRISM, a persistent intelligent assistant. "
    "You learn from every conversation and grow over time."
)


class SemanticStore:
    """
    Async interface to the semantic_memories table.

    Uses the same SQLite database as ExperienceLogger (shared db_url).
    Tables are created by ExperienceLogger.init(). Call that first.

    Args:
        db_url: SQLAlchemy async database URL. Defaults to config.EXPERIENCE_DB_URL.
    """

    def __init__(self, db_url: Optional[str] = None) -> None:
        self._db_url = db_url or config.EXPERIENCE_DB_URL
        if self._db_url.startswith("sqlite"):
            db_path_str = self._db_url.split("///")[-1]
            Path(db_path_str).parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_async_engine(self._db_url, echo=False)
        self._session_factory = async_sessionmaker(
            self._engine, expire_on_commit=False, class_=AsyncSession
        )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def create_semantic_memory(
        self,
        content: str,
        source_episode_ids: List[str],
        key_pattern: str,
        confidence: float,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Store a new provisional semantic memory.

        Args:
            content:             The compressed semantic memory text.
            source_episode_ids:  Episode UUIDs that were consolidated.
            key_pattern:         One-line summary of the pattern found.
            confidence:          LLM confidence at creation (0.0–1.0).
            metadata:            Optional extra metadata dict.

        Returns:
            The new semantic memory's UUID string.
        """
        mem_id = str(uuid.uuid4())
        row = SemanticMemoryRow(
            id=mem_id,
            content=content,
            source_episode_ids=json.dumps(source_episode_ids),
            key_pattern=key_pattern,
            confidence=confidence,
            is_provisional=True,
            validation_count=0,
            used_in_training=False,
            fitness_score=0.6,
            created_at=datetime.utcnow(),
            promoted_at=None,
            metadata_=json.dumps(metadata or {}),
        )
        async with self._session_factory() as session:
            session.add(row)
            await session.commit()
        logger.debug("Created provisional semantic memory %s", mem_id)
        return mem_id

    async def increment_validation(self, semantic_id: str) -> int:
        """
        Increment validation_count by 1.

        Returns:
            The new validation_count value.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(SemanticMemoryRow).where(SemanticMemoryRow.id == semantic_id)
            )
            row = result.scalar_one_or_none()
            if row is None:
                return 0
            row.validation_count += 1
            new_count = row.validation_count
            await session.commit()
        return new_count

    async def promote_to_confirmed(self, semantic_id: str) -> None:
        """Mark a provisional semantic memory as confirmed."""
        async with self._session_factory() as session:
            await session.execute(
                update(SemanticMemoryRow)
                .where(SemanticMemoryRow.id == semantic_id)
                .values(
                    is_provisional=False,
                    promoted_at=datetime.utcnow(),
                    fitness_score=1.0,
                )
            )
            await session.commit()
        logger.info("Promoted semantic memory %s to confirmed.", semantic_id)

    async def delete_semantic(self, semantic_id: str) -> None:
        """Hard-delete a semantic memory by ID."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(SemanticMemoryRow).where(SemanticMemoryRow.id == semantic_id)
            )
            row = result.scalar_one_or_none()
            if row is not None:
                await session.delete(row)
                await session.commit()
        logger.debug("Deleted semantic memory %s", semantic_id)

    async def mark_training_used(self, ids: List[str]) -> None:
        """Mark semantic memories as included in a LoRA training run."""
        if not ids:
            return
        async with self._session_factory() as session:
            await session.execute(
                update(SemanticMemoryRow)
                .where(SemanticMemoryRow.id.in_(ids))
                .values(used_in_training=True)
            )
            await session.commit()
        logger.info("Marked %d semantic memories as used_in_training=True", len(ids))

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def get_confirmed_semantics(self, limit: int = 100) -> List[dict]:
        """Return all confirmed (non-provisional) semantic memories, best first."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(SemanticMemoryRow)
                .where(SemanticMemoryRow.is_provisional == False)  # noqa: E712
                .order_by(SemanticMemoryRow.fitness_score.desc())
                .limit(limit)
            )
            rows = result.scalars().all()
        return [self._row_to_dict(r) for r in rows]

    async def get_provisional_semantics(self) -> List[dict]:
        """Return all provisional semantic memories (awaiting validation)."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(SemanticMemoryRow)
                .where(SemanticMemoryRow.is_provisional == True)  # noqa: E712
                .order_by(SemanticMemoryRow.created_at.asc())
            )
            rows = result.scalars().all()
        return [self._row_to_dict(r) for r in rows]

    async def get_training_ready_semantics(self) -> List[dict]:
        """Return confirmed semantics not yet used in training."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(SemanticMemoryRow)
                .where(SemanticMemoryRow.is_provisional == False)  # noqa: E712
                .where(SemanticMemoryRow.used_in_training == False)  # noqa: E712
                .order_by(SemanticMemoryRow.fitness_score.desc())
            )
            rows = result.scalars().all()
        return [self._row_to_dict(r) for r in rows]

    async def get_by_id(self, semantic_id: str) -> Optional[dict]:
        """Return a single semantic memory by ID, or None."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(SemanticMemoryRow).where(SemanticMemoryRow.id == semantic_id)
            )
            row = result.scalar_one_or_none()
        return self._row_to_dict(row) if row is not None else None

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_for_training(self, semantic: dict) -> str:
        """
        Format a confirmed semantic memory as a ChatML training example.

        The model is trained to respond naturally while having the semantic
        truth embedded in the system context.
        """
        return (
            f"<|im_start|>system\n"
            f"{_SYSTEM_PROMPT}\n"
            f"Confirmed truth about this user: {semantic['content']}"
            f"<|im_end|>\n"
            f"<|im_start|>user\n"
            f"[User continues conversation with context of above truth]"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"[Response informed by confirmed semantic memory: {semantic['key_pattern']}]"
            f"<|im_end|>"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_dict(row: SemanticMemoryRow) -> dict:
        return {
            "id": row.id,
            "content": row.content,
            "source_episode_ids": json.loads(row.source_episode_ids or "[]"),
            "key_pattern": row.key_pattern,
            "confidence": row.confidence,
            "is_provisional": row.is_provisional,
            "validation_count": row.validation_count,
            "used_in_training": row.used_in_training,
            "fitness_score": row.fitness_score,
            "created_at": row.created_at,
            "promoted_at": row.promoted_at,
            "metadata": json.loads(row.metadata_ or "{}"),
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_store_instance: Optional[SemanticStore] = None


def get_semantic_store() -> SemanticStore:
    """Return (or lazily create) the global SemanticStore."""
    global _store_instance
    if _store_instance is None:
        _store_instance = SemanticStore()
    return _store_instance
