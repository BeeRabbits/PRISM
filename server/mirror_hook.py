"""
MIRROR Post-Response Hook: fires after every chat response.

Runs oracle scoring and self-scoring in background, then updates
the episode record with both scores and the computed delta.
"""

from __future__ import annotations

import logging
from typing import Optional

import config
from data.experience_log import get_experience_logger
from server.mirror_oracle import oracle_score
from training.dream_consolidation import self_score

logger = logging.getLogger(__name__)


async def mirror_post_response(
    episode_id: str,
    user_message: str,
    prism_response: str,
    model=None,
    tokenizer=None,
) -> None:
    """
    Background task: score an episode via oracle + self-evaluation.

    Called as an asyncio background task after the response is sent.
    Never raises. All errors are caught and logged.
    """
    if not config.MIRROR_ENABLED:
        return

    if not episode_id:
        return

    exp_log = get_experience_logger()
    o_score: Optional[float] = None
    s_score: Optional[float] = None
    reasoning: Optional[str] = None

    # Oracle scoring (Claude API), skip if converged
    if not config.MIRROR_CONVERGED:
        try:
            result = await oracle_score(user_message, prism_response)
            if result is not None:
                o_score = result["oracle_score"]
                reasoning = result["reasoning"]
                logger.debug(
                    "MIRROR oracle: episode %s → %.1f (%s)",
                    episode_id, o_score, reasoning,
                )
        except Exception as e:
            logger.error("MIRROR oracle hook failed: %s", e)

    # Self-scoring (local model inference)
    if model is not None and tokenizer is not None:
        try:
            result = await self_score(model, tokenizer, user_message, prism_response)
            if result is not None:
                s_score = result["self_score"]
                logger.debug(
                    "MIRROR self-score: episode %s → %.1f",
                    episode_id, s_score,
                )
        except Exception as e:
            logger.error("MIRROR self-score hook failed: %s", e)

    # Update episode record
    if o_score is not None or s_score is not None:
        try:
            delta = await exp_log.update_mirror_scores(
                episode_id=episode_id,
                oracle_score=o_score,
                self_score=s_score,
                oracle_reasoning=reasoning,
            )
            if delta is not None:
                logger.info(
                    "MIRROR: episode %s scored, oracle=%.1f self=%.1f delta=%.2f",
                    episode_id,
                    o_score or -1,
                    s_score or -1,
                    delta,
                )
        except Exception as e:
            logger.error("MIRROR: failed to update episode %s: %s", episode_id, e)
