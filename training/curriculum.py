"""
Curriculum selector. chooses which episodes to include in a training run.

Currently implements a fitness-weighted selection strategy:
  - Episodes with fitness >= TRAINING_MIN_FITNESS are included.
  - Episodes are sorted by fitness descending (highest quality first).
  - A configurable cap limits the total training set size to avoid
    runaway memory and compute costs.

This module is a hook point for future curriculum learning strategies
(e.g. importance sampling, anti-curriculum, forgetting-based selection).
"""

from __future__ import annotations

import logging
from typing import List, Optional

import config
from data.experience_log import ExperienceLogger, get_experience_logger

logger = logging.getLogger(__name__)

# Hard cap on episodes per training run to bound compute cost
_MAX_EPISODES_PER_RUN = 5_000


class CurriculumSelector:
    """
    Selects training episodes based on fitness score.

    Args:
        experience_logger: The episode data source.
        min_fitness:       Minimum fitness score for inclusion.
        max_episodes:      Cap on total episodes returned.
    """

    def __init__(
        self,
        experience_logger: Optional[ExperienceLogger] = None,
        min_fitness: Optional[float] = None,
        max_episodes: int = _MAX_EPISODES_PER_RUN,
    ) -> None:
        self._log = experience_logger or get_experience_logger()
        self._min_fitness = min_fitness if min_fitness is not None else config.TRAINING_MIN_FITNESS
        self._max_episodes = max_episodes

    async def select(self) -> List[dict]:
        """
        Return a list of episodes to train on, sorted by fitness descending.

        Returns:
            List of episode dicts. Empty if not enough data.
        """
        candidates = await self._log.get_high_fitness_episodes(
            min_score=self._min_fitness,
            limit=self._max_episodes,
        )

        if len(candidates) < config.TRAINING_MIN_EPISODES:
            logger.info(
                "CurriculumSelector: only %d qualifying episodes (need %d). returning empty.",
                len(candidates),
                config.TRAINING_MIN_EPISODES,
            )
            return []

        # Sort by fitness descending (already done by the query, but be explicit)
        selected = sorted(candidates, key=lambda e: e["fitness_score"], reverse=True)
        logger.info(
            "CurriculumSelector: selected %d episodes (min_fitness=%.2f)",
            len(selected),
            self._min_fitness,
        )
        return selected
