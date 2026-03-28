"""
ActiveRecallLoop. tests model retention and re-queues forgotten episodes.

Neuroscience basis: Active recall (testing yourself) strengthens memory
traces more effectively than passive review. Episodes where the model
fails to reproduce the gist of the original response are "weak" and
get fast-tracked for replay in the next training batch.

Pipeline:
  1. Sample N episodes that have been trained on.
  2. Prompt the model with each user_message.
  3. Compare generated response to stored assistant_response via trigram cosine.
  4. Episodes scoring below ACTIVE_RECALL_THRESHOLD are considered "forgotten".
  5. Reset their replay schedule to immediate (next_replay_at = now).

Integrated into TrainingScheduler. runs before each training cycle.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, List, Optional, Tuple

import torch

import config
from data.experience_log import ExperienceLogger, get_experience_logger
from utils.text_similarity import cosine_similarity_trigram

logger = logging.getLogger(__name__)


class ActiveRecallLoop:
    """
    Tests model recall on previously trained episodes and identifies weak memories.

    Args:
        model:             The loaded causal LM.
        tokenizer:         Associated tokenizer.
        experience_logger: Source of episodes to test.
    """

    def __init__(
        self,
        model,
        tokenizer,
        experience_logger: Optional[ExperienceLogger] = None,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._log = experience_logger or get_experience_logger()

    async def run(self) -> dict:
        """
        Run one active recall test cycle.

        Returns:
            dict with total_tested, weak_count, avg_score, weak_episode_ids.
        """
        if not config.ACTIVE_RECALL_ENABLED:
            return {"status": "disabled"}

        # Sample trained episodes for recall testing
        episodes = await self._get_trained_sample()
        if not episodes:
            logger.info("ActiveRecall: no trained episodes to test.")
            return {"status": "skipped", "reason": "no_trained_episodes"}

        logger.info("ActiveRecall: testing recall on %d episodes.", len(episodes))

        results: List[Tuple[str, float]] = []
        weak_ids: List[str] = []

        for ep in episodes:
            score = self._test_single_recall(
                ep["user_message"],
                ep["assistant_response"],
            )
            results.append((ep["id"], score))
            if score < config.ACTIVE_RECALL_THRESHOLD:
                weak_ids.append(ep["id"])

        avg_score = sum(s for _, s in results) / len(results) if results else 0.0

        # Re-queue weak episodes for immediate replay
        if weak_ids:
            await self._log.reset_replay_schedule(weak_ids)
            logger.info(
                "ActiveRecall: %d/%d episodes weak (avg=%.3f). Re-queued for replay.",
                len(weak_ids), len(episodes), avg_score,
            )

        return {
            "status": "completed",
            "total_tested": len(episodes),
            "weak_count": len(weak_ids),
            "avg_score": round(avg_score, 4),
            "weak_episode_ids": weak_ids,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _get_trained_sample(self) -> List[dict]:
        """Get a random sample of previously trained episodes."""
        return await self._log.get_trained_episodes_sample(
            limit=config.ACTIVE_RECALL_SAMPLE_SIZE,
        )

    def _test_single_recall(
        self,
        user_message: str,
        original_response: str,
    ) -> float:
        """
        Prompt the model with user_message, generate a response,
        and score similarity to the original response.

        Returns:
            Cosine similarity score (0.0 to 1.0).
        """
        prompt = (
            "<|im_start|>system\n"
            "You are PRISM, a persistent intelligent assistant. "
            "You learn from every conversation and grow over time.\n"
            "<|im_end|>\n"
            f"<|im_start|>user\n{user_message}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        try:
            device = next(self._model.parameters()).device
            enc = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            input_ids = enc["input_ids"].to(device)

            with torch.no_grad():
                output = self._model.generate(
                    input_ids,
                    max_new_tokens=150,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )

            new_tokens = output[0][input_ids.shape[1]:]
            generated = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            return cosine_similarity_trigram(generated, original_response)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("ActiveRecall: OOM during recall test. skipping.")
                torch.cuda.empty_cache()
                return 0.5  # neutral score on OOM
            raise

