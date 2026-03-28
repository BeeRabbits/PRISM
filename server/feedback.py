"""
/feedback endpoint: accepts fitness signals for logged episodes.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from data.experience_log import get_experience_logger
from data.schemas import FeedbackRequest, FeedbackResponse

logger = logging.getLogger(__name__)

router = APIRouter()


async def feedback_endpoint(request: FeedbackRequest) -> FeedbackResponse:
    """
    POST /feedback

    Updates the fitness score for an episode:
      - score < 0.2: flags the episode (excluded from training automatically
                     because DatasetBuilder filters on min_fitness).
      - score > 0.8: high-priority. The EMA update will push fitness high.
    """
    exp_log = get_experience_logger()

    try:
        new_fitness = await exp_log.update_fitness(
            episode_id=request.episode_id,
            score=request.score,
            signal_type="explicit",
            reason=request.reason,
            was_useful=request.was_useful,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    logger.info(
        "Feedback recorded for episode %s: score=%.2f new_fitness=%.3f",
        request.episode_id,
        request.score,
        new_fitness,
    )

    return FeedbackResponse(
        episode_id=request.episode_id,
        new_fitness=new_fitness,
    )
