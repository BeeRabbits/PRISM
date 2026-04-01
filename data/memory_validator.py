"""
Memory Validator: "hint, not truth" pattern for semantic memories.

Inspired by Claude Code's memory validation system that treats stored
memories as hints requiring verification before use.

Before injecting semantic memories into responses, this module validates:
  1. Source episode fitness (are the originating episodes still high-quality?)
  2. Staleness (has this memory been validated recently?)
  3. Confidence threshold (is confidence high enough to trust?)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

import config

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating a single semantic memory."""
    is_valid: bool
    reason: str
    adjusted_confidence: float


class MemoryValidator:
    """
    Validates semantic memories before they are injected into
    the inference system prompt.

    Args:
        experience_logger: The ExperienceLogger for checking source episodes.
        semantic_store:    The SemanticStore for updating validation timestamps.
    """

    def __init__(self, experience_logger=None, semantic_store=None) -> None:
        self._log = experience_logger
        self._store = semantic_store

    async def validate_memory(self, memory: dict) -> ValidationResult:
        """
        Validate a single semantic memory.

        Checks:
          1. Confidence above minimum threshold
          2. Staleness (last_validated_at within max days)
          3. Source episode fitness (average fitness still acceptable)

        Returns:
            ValidationResult with is_valid, reason, and adjusted_confidence.
        """
        confidence = memory.get("confidence", 0.0)

        # Check 1: Confidence threshold
        if confidence < config.MEMORY_VALIDATION_MIN_CONFIDENCE:
            return ValidationResult(
                is_valid=False,
                reason=f"confidence {confidence:.2f} below threshold {config.MEMORY_VALIDATION_MIN_CONFIDENCE}",
                adjusted_confidence=confidence,
            )

        # Check 2: Staleness
        last_validated = memory.get("last_validated_at")
        if last_validated is not None:
            if isinstance(last_validated, str):
                last_validated = datetime.fromisoformat(last_validated)
            max_age = timedelta(days=config.MEMORY_VALIDATION_MAX_STALENESS_DAYS)
            if datetime.utcnow() - last_validated > max_age:
                # Stale but not invalid. Reduce confidence
                adjusted = confidence * 0.7
                return ValidationResult(
                    is_valid=adjusted >= config.MEMORY_VALIDATION_MIN_CONFIDENCE,
                    reason=f"stale (last validated {last_validated}), confidence adjusted to {adjusted:.2f}",
                    adjusted_confidence=adjusted,
                )

        # Check 3: Source episode fitness (if logger available)
        if self._log is not None:
            source_ids = memory.get("source_episode_ids", [])
            if isinstance(source_ids, str):
                source_ids = json.loads(source_ids)
            if source_ids:
                avg_fitness = await self._check_source_fitness(source_ids)
                if avg_fitness is not None and avg_fitness < config.MEMORY_VALIDATION_MIN_SOURCE_FITNESS:
                    return ValidationResult(
                        is_valid=False,
                        reason=f"source episodes avg fitness {avg_fitness:.2f} below threshold",
                        adjusted_confidence=confidence * 0.5,
                    )

        return ValidationResult(
            is_valid=True,
            reason="valid",
            adjusted_confidence=confidence,
        )

    async def validate_batch(self, memories: List[dict]) -> List[dict]:
        """
        Validate a batch of semantic memories and return only valid ones.

        Each returned memory gets an additional 'validation_status' field.
        Results are sorted by adjusted confidence (highest first).
        """
        validated = []
        for mem in memories:
            result = await self.validate_memory(mem)
            if result.is_valid:
                mem["validation_status"] = result.reason
                mem["adjusted_confidence"] = result.adjusted_confidence
                validated.append(mem)
            else:
                logger.debug(
                    "Memory %s failed validation: %s",
                    mem.get("id", "unknown"), result.reason,
                )

        # Sort by adjusted confidence descending
        validated.sort(key=lambda m: m.get("adjusted_confidence", 0), reverse=True)
        return validated

    async def _check_source_fitness(self, episode_ids: List[str]) -> Optional[float]:
        """Check average fitness of source episodes."""
        try:
            from sqlalchemy import select
            from data.experience_log import EpisodeRow

            async with self._log._session_factory() as session:
                result = await session.execute(
                    select(EpisodeRow.fitness_score)
                    .where(EpisodeRow.id.in_(episode_ids))
                )
                scores = [row[0] for row in result.all() if row[0] is not None]

            if not scores:
                return None
            return sum(scores) / len(scores)
        except Exception:
            logger.debug("Could not check source fitness, skipping check.")
            return None


# Module-level singleton
_validator_instance: Optional[MemoryValidator] = None


def get_memory_validator() -> MemoryValidator:
    """Return (or lazily create) the global MemoryValidator."""
    global _validator_instance
    if _validator_instance is None:
        from data.experience_log import get_experience_logger
        from data.semantic_store import get_semantic_store
        _validator_instance = MemoryValidator(
            experience_logger=get_experience_logger(),
            semantic_store=get_semantic_store(),
        )
    return _validator_instance
