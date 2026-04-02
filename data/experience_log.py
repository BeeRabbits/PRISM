"""
ExperienceLogger: async SQLite logger for all interaction episodes.

Tables:
  episodes       - one row per chat turn
  fitness_signals - feedback signals attached to episodes

Uses SQLAlchemy async with aiosqlite.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    select,
    text,
    update,
)
from datetime import timedelta
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ORM models
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class EpisodeRow(Base):
    __tablename__ = "episodes"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    session_id = Column(String(255), nullable=False, index=True)
    user_message = Column(Text, nullable=False)
    assistant_response = Column(Text, nullable=False)
    full_prompt = Column(Text, nullable=False)
    memory_context_used = Column(Text, default="")
    response_quality = Column(Float, nullable=True)
    was_memory_useful = Column(Boolean, nullable=True)
    training_used = Column(Boolean, default=False, nullable=False)
    fitness_score = Column(Float, default=1.0, nullable=False)
    memory_type = Column(String(64), default="episodic", nullable=False)
    used_in_consolidation = Column(Boolean, default=False, nullable=False)
    # MIRROR Protocol fields
    oracle_score = Column(Float, nullable=True)
    self_score = Column(Float, nullable=True)
    mirror_delta = Column(Float, nullable=True)
    mirror_iteration = Column(Integer, default=0, nullable=False)
    auto_scored = Column(Boolean, default=False, nullable=False)
    oracle_reasoning = Column(Text, nullable=True)
    # Spaced Repetition fields
    replay_count = Column(Integer, default=0, nullable=False)
    next_replay_at = Column(DateTime, nullable=True)
    replay_interval_days = Column(Integer, default=1, nullable=False)
    last_recall_score = Column(Float, nullable=True)
    # Frustration Detection fields
    sentiment_tier = Column(String(32), nullable=True)
    sentiment_patterns = Column(Text, nullable=True)


class SemanticMemoryRow(Base):
    __tablename__ = "semantic_memories"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    content = Column(Text, nullable=False)
    source_episode_ids = Column(Text, nullable=False)   # JSON array of UUIDs
    key_pattern = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False)
    is_provisional = Column(Boolean, default=True, nullable=False)
    validation_count = Column(Integer, default=0, nullable=False)
    used_in_training = Column(Boolean, default=False, nullable=False)
    fitness_score = Column(Float, default=0.6, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    promoted_at = Column(DateTime, nullable=True)
    last_validated_at = Column(DateTime, nullable=True)
    metadata_ = Column("metadata", Text, default="{}")


class ContradictionLogRow(Base):
    __tablename__ = "contradiction_log"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    episode_a_id = Column(String(36), nullable=False, index=True)
    episode_b_id = Column(String(36), nullable=True)
    resolution = Column(String(64), nullable=False)
    reason = Column(Text, default="")
    confidence = Column(Float, nullable=False)
    resolved_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class FitnessSignalRow(Base):
    __tablename__ = "fitness_signals"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    episode_id = Column(String(36), ForeignKey("episodes.id"), nullable=False, index=True)
    signal_type = Column(String(64), nullable=False)  # explicit | implicit | contradiction
    score = Column(Float, nullable=False)
    reason = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


# ---------------------------------------------------------------------------
# ExperienceLogger
# ---------------------------------------------------------------------------

class ExperienceLogger:
    """
    Async logger for chat episodes and fitness signals.

    Usage:
        logger = ExperienceLogger()
        await logger.init()
        episode_id = await logger.log_episode(...)
        await logger.update_fitness(episode_id, score=0.9, ...)
    """

    def __init__(self, db_url: Optional[str] = None) -> None:
        self._db_url = db_url or config.EXPERIENCE_DB_URL
        # Ensure the data directory exists for SQLite
        if self._db_url.startswith("sqlite"):
            db_path_str = self._db_url.split("///")[-1]
            Path(db_path_str).parent.mkdir(parents=True, exist_ok=True)

        self._engine = create_async_engine(
            self._db_url,
            echo=False,
            connect_args={"timeout": 30},  # wait up to 30s for DB lock
        )
        self._session_factory = async_sessionmaker(
            self._engine, expire_on_commit=False, class_=AsyncSession
        )

    async def init(self) -> None:
        """Create tables if they don't exist, then apply incremental migrations."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            # Enable WAL mode for concurrent read/write support
            await conn.execute(text("PRAGMA journal_mode=WAL"))
            await conn.execute(text("PRAGMA busy_timeout=30000"))
        await self._migrate_schema()
        logger.info("ExperienceLogger initialised at %s", self._db_url)

    async def _migrate_schema(self) -> None:
        """Apply incremental column additions to existing tables (safe to re-run)."""
        migrations = [
            "ALTER TABLE episodes ADD COLUMN memory_type TEXT NOT NULL DEFAULT 'episodic'",
            "ALTER TABLE episodes ADD COLUMN used_in_consolidation BOOLEAN NOT NULL DEFAULT 0",
            # MIRROR Protocol columns
            "ALTER TABLE episodes ADD COLUMN oracle_score REAL",
            "ALTER TABLE episodes ADD COLUMN self_score REAL",
            "ALTER TABLE episodes ADD COLUMN mirror_delta REAL",
            "ALTER TABLE episodes ADD COLUMN mirror_iteration INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE episodes ADD COLUMN auto_scored BOOLEAN NOT NULL DEFAULT 0",
            "ALTER TABLE episodes ADD COLUMN oracle_reasoning TEXT",
            # Spaced Repetition columns
            "ALTER TABLE episodes ADD COLUMN replay_count INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE episodes ADD COLUMN next_replay_at DATETIME",
            "ALTER TABLE episodes ADD COLUMN replay_interval_days INTEGER NOT NULL DEFAULT 1",
            "ALTER TABLE episodes ADD COLUMN last_recall_score REAL",
            # Frustration Detection columns
            "ALTER TABLE episodes ADD COLUMN sentiment_tier TEXT",
            "ALTER TABLE episodes ADD COLUMN sentiment_patterns TEXT",
            # Memory Validation column
            "ALTER TABLE semantic_memories ADD COLUMN last_validated_at DATETIME",
        ]
        async with self._engine.begin() as conn:
            for sql in migrations:
                try:
                    await conn.execute(text(sql))
                except Exception:
                    pass  # Column already exists. Expected on repeat runs

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def log_episode(
        self,
        user_message: str,
        assistant_response: str,
        full_prompt: str,
        session_id: str,
        memory_context_used: str = "",
        sentiment_tier: Optional[str] = None,
        sentiment_patterns: Optional[str] = None,
    ) -> str:
        """
        Log a new episode and return its UUID string.

        Args:
            user_message:        The raw user input.
            assistant_response:  The model's reply.
            full_prompt:         The full prompt string sent to the model.
            session_id:          Session identifier.
            memory_context_used: Any memory context injected into the prompt.
            sentiment_tier:      Detected frustration tier (MILD/FRUSTRATED/ANGRY).
            sentiment_patterns:  JSON array of matched frustration pattern names.

        Returns:
            episode_id as a string UUID.
        """
        episode_id = str(uuid.uuid4())
        row = EpisodeRow(
            id=episode_id,
            session_id=session_id,
            user_message=user_message,
            assistant_response=assistant_response,
            full_prompt=full_prompt,
            memory_context_used=memory_context_used,
            timestamp=datetime.utcnow(),
            sentiment_tier=sentiment_tier,
            sentiment_patterns=sentiment_patterns,
        )
        async with self._session_factory() as session:
            session.add(row)
            await session.commit()
        logger.debug("Logged episode %s for session %s", episode_id, session_id)
        return episode_id

    async def update_fitness(
        self,
        episode_id: str,
        score: float,
        signal_type: str,
        reason: str = "",
        was_useful: Optional[bool] = None,
    ) -> float:
        """
        Update the fitness score for an episode and record the signal.

        The fitness score is updated as an EMA with the new score:
            new_fitness = 0.7 * old_fitness + 0.3 * score

        If the episode hasn't been scored yet (fitness_score == 1.0, the default),
        the score is set directly.

        Returns the updated fitness score.
        """
        async with self._session_factory() as session:
            # Fetch current fitness
            result = await session.execute(
                select(EpisodeRow).where(EpisodeRow.id == episode_id)
            )
            row = result.scalar_one_or_none()
            if row is None:
                raise ValueError(f"Episode not found: {episode_id}")

            # EMA update (or direct set if never scored)
            if row.response_quality is None:
                new_fitness = score
            else:
                new_fitness = 0.7 * row.fitness_score + 0.3 * score

            row.fitness_score = new_fitness
            row.response_quality = score
            if was_useful is not None:
                row.was_memory_useful = was_useful

            # Record fitness signal
            signal = FitnessSignalRow(
                episode_id=episode_id,
                signal_type=signal_type,
                score=score,
                reason=reason,
                created_at=datetime.utcnow(),
            )
            session.add(signal)
            await session.commit()

        logger.debug("Updated fitness for episode %s → %.3f", episode_id, new_fitness)
        return new_fitness

    async def update_mirror_scores(
        self,
        episode_id: str,
        oracle_score: Optional[float] = None,
        self_score: Optional[float] = None,
        oracle_reasoning: Optional[str] = None,
        mirror_iteration: int = 0,
    ) -> Optional[float]:
        """
        Update MIRROR scores for an episode and compute the delta.

        Also updates fitness_score from the oracle score (mapped from 1-5 to 0-1).
        Returns the computed delta, or None if both scores are not available.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(EpisodeRow).where(EpisodeRow.id == episode_id)
            )
            row = result.scalar_one_or_none()
            if row is None:
                logger.warning("MIRROR: episode not found: %s", episode_id)
                return None

            if oracle_score is not None:
                row.oracle_score = oracle_score
            if self_score is not None:
                row.self_score = self_score
            if oracle_reasoning is not None:
                row.oracle_reasoning = oracle_reasoning

            row.mirror_iteration = mirror_iteration
            row.auto_scored = True

            # Compute delta if both scores are available
            o = oracle_score if oracle_score is not None else row.oracle_score
            s = self_score if self_score is not None else row.self_score
            delta = None
            if o is not None and s is not None:
                delta = abs(o - s)
                row.mirror_delta = delta

            # MIRROR fitness strategy: keep fitness at default (1.0) so all
            # episodes are eligible for training. The DELTA drives learning
            # priority via weighted repetition in the data builder. Low oracle
            # scores mean high delta, which means MORE training weight, not less.
            # Only manual /feedback should lower fitness below the threshold.

            await session.commit()

        logger.debug(
            "MIRROR scores updated for episode %s: oracle=%.1f self=%.1f delta=%s",
            episode_id,
            oracle_score or -1,
            self_score or -1,
            f"{delta:.2f}" if delta is not None else "N/A",
        )
        return delta

    async def get_mirror_stats(self) -> dict:
        """Return aggregate MIRROR statistics for convergence monitoring."""
        async with self._session_factory() as session:
            # Count auto-scored episodes
            result = await session.execute(
                select(EpisodeRow).where(EpisodeRow.auto_scored == True)  # noqa: E712
            )
            auto_scored = result.scalars().all()

            # Get last N deltas for convergence window
            result2 = await session.execute(
                select(EpisodeRow)
                .where(EpisodeRow.mirror_delta.isnot(None))
                .order_by(EpisodeRow.timestamp.desc())
                .limit(config.MIRROR_CONVERGENCE_WINDOW)
            )
            recent = result2.scalars().all()

        deltas = [r.mirror_delta for r in recent if r.mirror_delta is not None]
        avg_delta = sum(deltas) / len(deltas) if deltas else None

        return {
            "auto_scored_count": len(auto_scored),
            "oracle_calls": len([e for e in auto_scored if e.oracle_score is not None]),
            "rolling_avg_delta": round(avg_delta, 3) if avg_delta is not None else None,
            "delta_sample_size": len(deltas),
        }

    async def get_episodes_with_delta(
        self,
        min_score: float,
        limit: int = 1000,
    ) -> List[dict]:
        """Return untrained high-fitness episodes including MIRROR delta info."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(EpisodeRow)
                .where(EpisodeRow.fitness_score >= min_score)
                .where(EpisodeRow.training_used == False)  # noqa: E712
                .order_by(EpisodeRow.fitness_score.desc())
                .limit(limit)
            )
            rows = result.scalars().all()

        return [
            {
                "id": r.id,
                "session_id": r.session_id,
                "user_message": r.user_message,
                "assistant_response": r.assistant_response,
                "fitness_score": r.fitness_score,
                "memory_context_used": r.memory_context_used,
                "timestamp": r.timestamp,
                "mirror_delta": r.mirror_delta,
                "oracle_score": r.oracle_score,
                "self_score": r.self_score,
            }
            for r in rows
        ]

    async def mark_training_used(self, episode_ids: List[str]) -> None:
        """Mark episodes as used in a training run."""
        if not episode_ids:
            return
        async with self._session_factory() as session:
            await session.execute(
                update(EpisodeRow)
                .where(EpisodeRow.id.in_(episode_ids))
                .values(training_used=True)
            )
            await session.commit()
        logger.info("Marked %d episodes as training_used=True", len(episode_ids))

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def get_high_fitness_episodes(
        self,
        min_score: float,
        limit: int = 1000,
    ) -> List[dict]:
        """
        Return high-fitness episodes not yet used in training.

        Args:
            min_score: Minimum fitness_score to include.
            limit:     Maximum number of episodes to return.

        Returns:
            List of dicts with keys matching EpisodeRow columns.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(EpisodeRow)
                .where(EpisodeRow.fitness_score >= min_score)
                .where(EpisodeRow.training_used == False)  # noqa: E712
                .order_by(EpisodeRow.fitness_score.desc())
                .limit(limit)
            )
            rows = result.scalars().all()

        return [
            {
                "id": r.id,
                "session_id": r.session_id,
                "user_message": r.user_message,
                "assistant_response": r.assistant_response,
                "fitness_score": r.fitness_score,
                "memory_context_used": r.memory_context_used,
                "timestamp": r.timestamp,
            }
            for r in rows
        ]

    async def get_all_high_fitness_episodes(
        self,
        min_score: float,
        limit: int = 1000,
    ) -> List[dict]:
        """Return high-fitness episodes regardless of training_used flag."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(EpisodeRow)
                .where(EpisodeRow.fitness_score >= min_score)
                .order_by(EpisodeRow.fitness_score.desc())
                .limit(limit)
            )
            rows = result.scalars().all()

        return [
            {
                "id": r.id,
                "session_id": r.session_id,
                "user_message": r.user_message,
                "assistant_response": r.assistant_response,
                "fitness_score": r.fitness_score,
                "memory_context_used": r.memory_context_used,
                "timestamp": r.timestamp,
            }
            for r in rows
        ]

    async def get_untrained_episodes(self, min_score: float) -> List[dict]:
        """Alias for get_high_fitness_episodes. Returns untrained episodes above min_score."""
        return await self.get_high_fitness_episodes(min_score=min_score)

    async def count_total(self) -> int:
        """Return total number of episodes."""
        async with self._session_factory() as session:
            result = await session.execute(select(EpisodeRow))
            return len(result.scalars().all())

    async def count_high_fitness(self, min_score: Optional[float] = None) -> int:
        """Return number of untrained high-fitness episodes."""
        threshold = min_score if min_score is not None else config.TRAINING_MIN_FITNESS
        episodes = await self.get_high_fitness_episodes(min_score=threshold)
        return len(episodes)

    async def get_consolidation_candidates(
        self,
        min_fitness: float = 0.6,
        hours_old: int = 48,
        limit: int = 500,
    ) -> List[dict]:
        """
        Return episodes eligible for dream consolidation:
        high-fitness, episodic, older than hours_old, not yet consolidated.
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours_old)
        async with self._session_factory() as session:
            result = await session.execute(
                select(EpisodeRow)
                .where(EpisodeRow.fitness_score >= min_fitness)
                .where(EpisodeRow.used_in_consolidation == False)  # noqa: E712
                .where(EpisodeRow.training_used == False)          # noqa: E712
                .where(EpisodeRow.timestamp < cutoff)
                .order_by(EpisodeRow.timestamp.asc())
                .limit(limit)
            )
            rows = result.scalars().all()
        return [
            {
                "id": r.id,
                "session_id": r.session_id,
                "user_message": r.user_message,
                "assistant_response": r.assistant_response,
                "fitness_score": r.fitness_score,
                "timestamp": r.timestamp,
                "mirror_delta": r.mirror_delta,
                "oracle_score": r.oracle_score,
            }
            for r in rows
        ]

    async def mark_consolidation_used(self, episode_ids: List[str]) -> None:
        """Mark episodes as consumed by dream consolidation."""
        if not episode_ids:
            return
        async with self._session_factory() as session:
            await session.execute(
                update(EpisodeRow)
                .where(EpisodeRow.id.in_(episode_ids))
                .values(used_in_consolidation=True)
            )
            await session.commit()
        logger.info("Marked %d episodes as used_in_consolidation=True", len(episode_ids))

    async def get_recent_episodes(
        self,
        days: int = 7,
        min_score: float = 0.5,
        session_id: Optional[str] = None,
        limit: int = 200,
    ) -> List[dict]:
        """
        Return recent episodes for contradiction checking and semantic validation.

        Args:
            days:       Look-back window in days.
            min_score:  Minimum fitness score to include.
            session_id: If set, filter to this session only.
            limit:      Maximum rows returned.
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        async with self._session_factory() as session:
            q = (
                select(EpisodeRow)
                .where(EpisodeRow.fitness_score >= min_score)
                .where(EpisodeRow.timestamp >= cutoff)
                .order_by(EpisodeRow.timestamp.desc())
                .limit(limit)
            )
            if session_id is not None:
                q = q.where(EpisodeRow.session_id == session_id)
            result = await session.execute(q)
            rows = result.scalars().all()
        return [
            {
                "id": r.id,
                "session_id": r.session_id,
                "user_message": r.user_message,
                "assistant_response": r.assistant_response,
                "fitness_score": r.fitness_score,
                "timestamp": r.timestamp,
            }
            for r in rows
        ]

    async def set_episode_fitness(self, episode_id: str, score: float) -> None:
        """Directly overwrite an episode's fitness_score (used by contradiction resolution)."""
        async with self._session_factory() as session:
            await session.execute(
                update(EpisodeRow)
                .where(EpisodeRow.id == episode_id)
                .values(fitness_score=score)
            )
            await session.commit()
        logger.debug("Set fitness for episode %s → %.3f", episode_id, score)

    async def log_contradiction(
        self,
        episode_a_id: str,
        episode_b_id: Optional[str],
        resolution: str,
        reason: str,
        confidence: float,
    ) -> str:
        """Log a detected contradiction to the contradiction_log table."""
        row_id = str(uuid.uuid4())
        row = ContradictionLogRow(
            id=row_id,
            episode_a_id=episode_a_id,
            episode_b_id=episode_b_id,
            resolution=resolution,
            reason=reason,
            confidence=confidence,
            resolved_at=datetime.utcnow(),
        )
        async with self._session_factory() as session:
            session.add(row)
            await session.commit()
        logger.debug("Logged contradiction %s (resolution=%s)", row_id, resolution)
        return row_id

    # ------------------------------------------------------------------
    # Spaced Repetition
    # ------------------------------------------------------------------

    async def init_replay_schedule(self, episode_ids: List[str]) -> None:
        """
        Initialize the replay schedule for newly trained episodes.
        Sets next_replay_at to 1 day from now (first Fibonacci interval).
        Called after mark_training_used().
        """
        if not episode_ids:
            return
        next_at = datetime.utcnow() + timedelta(days=1)
        async with self._session_factory() as session:
            await session.execute(
                update(EpisodeRow)
                .where(EpisodeRow.id.in_(episode_ids))
                .where(EpisodeRow.next_replay_at.is_(None))
                .values(
                    next_replay_at=next_at,
                    replay_count=0,
                    replay_interval_days=1,
                )
            )
            await session.commit()
        logger.info("Initialized replay schedule for %d episodes (next=%s).", len(episode_ids), next_at)

    async def get_due_episodes(self, limit: int = 200) -> List[dict]:
        """
        Return episodes whose next_replay_at has passed. They are due
        for spaced repetition review.
        """
        now = datetime.utcnow()
        async with self._session_factory() as session:
            result = await session.execute(
                select(EpisodeRow)
                .where(EpisodeRow.next_replay_at.isnot(None))
                .where(EpisodeRow.next_replay_at <= now)
                .order_by(EpisodeRow.next_replay_at.asc())
                .limit(limit)
            )
            rows = result.scalars().all()
        return [
            {
                "id": r.id,
                "session_id": r.session_id,
                "user_message": r.user_message,
                "assistant_response": r.assistant_response,
                "fitness_score": r.fitness_score,
                "replay_count": r.replay_count,
                "next_replay_at": r.next_replay_at,
                "mirror_delta": r.mirror_delta,
                "oracle_score": r.oracle_score,
                "self_score": r.self_score,
                "last_recall_score": r.last_recall_score,
            }
            for r in rows
        ]

    async def advance_replay_schedule(self, episode_ids: List[str]) -> None:
        """
        Advance the Fibonacci replay schedule for episodes that have been
        successfully reviewed. Each call increments replay_count and sets
        the next interval from the Fibonacci sequence.

        Fibonacci intervals (days): 1, 1, 2, 3, 5, 8, 13, 21, 34, 55
        After exhausting the sequence, the episode is "graduated" and gets no more replays.
        """
        if not episode_ids:
            return
        fib = config.SPACED_REPLAY_FIBONACCI
        async with self._session_factory() as session:
            for eid in episode_ids:
                result = await session.execute(
                    select(EpisodeRow).where(EpisodeRow.id == eid)
                )
                row = result.scalar_one_or_none()
                if row is None:
                    continue

                new_count = row.replay_count + 1
                if new_count >= len(fib):
                    # Graduated. No more replays needed
                    row.replay_count = new_count
                    row.next_replay_at = None
                    logger.debug("Episode %s graduated from spaced replay.", eid)
                else:
                    interval = fib[new_count]
                    row.replay_count = new_count
                    row.replay_interval_days = interval
                    row.next_replay_at = datetime.utcnow() + timedelta(days=interval)
                    logger.debug(
                        "Episode %s advanced to replay %d (next in %d days).",
                        eid, new_count, interval,
                    )
            await session.commit()
        logger.info("Advanced replay schedule for %d episodes.", len(episode_ids))

    async def reset_replay_schedule(self, episode_ids: List[str]) -> None:
        """
        Reset episodes to immediate replay (next_replay_at = now).
        Called by ActiveRecallLoop when an episode is forgotten.
        Also resets the interval back to 1 day.
        """
        if not episode_ids:
            return
        now = datetime.utcnow()
        async with self._session_factory() as session:
            await session.execute(
                update(EpisodeRow)
                .where(EpisodeRow.id.in_(episode_ids))
                .values(
                    next_replay_at=now,
                    replay_interval_days=1,
                    # Don't reset replay_count. Preserve history
                )
            )
            await session.commit()
        logger.info("Reset replay schedule for %d weak episodes to immediate.", len(episode_ids))

    async def update_recall_score(self, episode_id: str, score: float) -> None:
        """Store the latest active recall score for an episode."""
        async with self._session_factory() as session:
            await session.execute(
                update(EpisodeRow)
                .where(EpisodeRow.id == episode_id)
                .values(last_recall_score=score)
            )
            await session.commit()

    async def get_trained_episodes_sample(self, limit: int = 30) -> List[dict]:
        """
        Return a random sample of previously trained episodes for recall testing.
        Prioritizes episodes with active replay schedules.
        """
        async with self._session_factory() as session:
            # First get episodes with active replay schedules
            result = await session.execute(
                select(EpisodeRow)
                .where(EpisodeRow.training_used == True)  # noqa: E712
                .where(EpisodeRow.next_replay_at.isnot(None))
                .order_by(EpisodeRow.next_replay_at.asc())
                .limit(limit)
            )
            rows = list(result.scalars().all())

            # If not enough, supplement with any trained episodes
            if len(rows) < limit:
                existing_ids = [r.id for r in rows]
                result2 = await session.execute(
                    select(EpisodeRow)
                    .where(EpisodeRow.training_used == True)  # noqa: E712
                    .order_by(EpisodeRow.timestamp.desc())
                    .limit(limit * 2)
                )
                extra = [r for r in result2.scalars().all() if r.id not in existing_ids]
                import random
                random.shuffle(extra)
                rows.extend(extra[: limit - len(rows)])

        return [
            {
                "id": r.id,
                "session_id": r.session_id,
                "user_message": r.user_message,
                "assistant_response": r.assistant_response,
                "fitness_score": r.fitness_score,
                "replay_count": r.replay_count,
                "mirror_delta": r.mirror_delta,
            }
            for r in rows
        ]

    async def get_contradictions(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> List[dict]:
        """Return recent contradiction log entries."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(ContradictionLogRow)
                .order_by(ContradictionLogRow.resolved_at.desc())
                .limit(limit)
                .offset(offset)
            )
            rows = result.scalars().all()
        return [
            {
                "id": r.id,
                "episode_a_id": r.episode_a_id,
                "episode_b_id": r.episode_b_id,
                "resolution": r.resolution,
                "reason": r.reason,
                "confidence": r.confidence,
                "resolved_at": r.resolved_at,
            }
            for r in rows
        ]


# Module-level singleton
_logger_instance: Optional[ExperienceLogger] = None


def get_experience_logger() -> ExperienceLogger:
    """Return (or lazily create) the global ExperienceLogger."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = ExperienceLogger()
    return _logger_instance
