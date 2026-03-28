"""
ContradictionEngine. real-time contradiction detection and resolution.

Runs before every new episode is logged to keep training data clean.

Pipeline:
  1. Compute text similarity between the new episode and recent episodes
     for the same user (last 90 days, fitness >= 0.5).
  2. For the top matches above CONTRADICTION_SIMILARITY_THRESHOLD, call
     Claude to determine if a real contradiction exists.
  3. Apply the recommended resolution:
       keep_a  → block the new episode entirely
       keep_b  → zero out the existing episode's fitness, store new
       merge   → zero out existing, store merged content
       keep_both → store new alongside existing (no conflict)
  4. Log every contradiction decision to contradiction_log.

All LLM failures degrade gracefully. if Claude is unavailable, the episode
is stored normally and the check is skipped.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import config
from data.experience_log import ExperienceLogger, get_experience_logger
from utils.text_similarity import cosine_similarity_trigram

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ResolutionDetail:
    """Details about a single contradiction resolution."""
    existing_episode_id: str
    resolution: str        # keep_a | keep_b | merge | keep_both
    confidence: float
    reason: str
    merged_content: Optional[str] = None


@dataclass
class ContradictionResult:
    """
    Full result from check_and_resolve().

    Attributes:
        contradictions_found: Number of genuine contradictions detected.
        resolutions:          Details for each resolved contradiction.
        final_action:         "stored" | "blocked" | "merged"
        final_content:        Merged content if final_action == "merged".
    """
    contradictions_found: int = 0
    resolutions: List[ResolutionDetail] = field(default_factory=list)
    final_action: str = "stored"
    final_content: Optional[str] = None


# ---------------------------------------------------------------------------
# ContradictionEngine
# ---------------------------------------------------------------------------

class ContradictionEngine:
    """
    Detects and resolves contradictions between a new episode and recent
    high-fitness episodes for the same user.

    Args:
        experience_logger: Used to query and update episode fitness.
        anthropic_client:  anthropic.AsyncAnthropic instance, or None for
                           lazy creation from config.ANTHROPIC_API_KEY.
    """

    def __init__(
        self,
        experience_logger: Optional[ExperienceLogger] = None,
        anthropic_client: Optional[Any] = None,
    ) -> None:
        self._log = experience_logger or get_experience_logger()
        self._client = anthropic_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def check_and_resolve(
        self,
        new_episode_content: str,
        user_id: str,
    ) -> ContradictionResult:
        """
        Check a new episode for contradictions against the user's history.

        Args:
            new_episode_content: The full conversation turn text (user + assistant).
            user_id:             The session_id / user identifier.

        Returns:
            ContradictionResult with the final_action to take.
        """
        client = self._get_client()
        if client is None:
            # No Anthropic access. store normally, skip check
            return ContradictionResult(final_action="stored")

        # Step 1. Fetch recent episodes for this user
        try:
            recent = await self._log.get_recent_episodes(
                days=90,
                min_score=0.5,
                session_id=user_id,
                limit=200,
            )
        except Exception as e:
            logger.error("ContradictionEngine: DB query failed: %s", e)
            return ContradictionResult(final_action="stored")

        if not recent:
            return ContradictionResult(final_action="stored")

        # Step 2. Find high-similarity candidates
        threshold = config.CONTRADICTION_SIMILARITY_THRESHOLD
        candidates: List[dict] = []

        for ep in recent:
            ep_text = f"User: {ep['user_message']}\nAssistant: {ep['assistant_response']}"
            sim = cosine_similarity_trigram(new_episode_content, ep_text)
            if sim >= threshold:
                candidates.append({**ep, "_similarity": sim})

        # Sort by similarity descending, check top 10
        candidates.sort(key=lambda x: x["_similarity"], reverse=True)
        candidates = candidates[:10]

        if not candidates:
            return ContradictionResult(final_action="stored")

        # Step 3-4. Check each candidate for contradiction
        result = ContradictionResult()
        final_content = new_episode_content

        for existing in candidates:
            existing_text = (
                f"User: {existing['user_message']}\n"
                f"Assistant: {existing['assistant_response']}"
            )
            resolution = await self._check_pair(
                existing_episode=existing,
                existing_text=existing_text,
                new_text=new_episode_content,
                client=client,
            )
            if resolution is None:
                continue

            result.contradictions_found += 1
            result.resolutions.append(resolution)

            # Apply resolution
            if resolution.resolution == "keep_a":
                # Block new episode. existing wins
                result.final_action = "blocked"
                logger.info(
                    "ContradictionEngine: blocking new episode. existing %s wins.",
                    existing["id"],
                )
                # Log and return immediately. first keep_a wins
                await self._log_contradiction(existing["id"], None, "keep_a", resolution)
                return result

            elif resolution.resolution == "keep_b":
                # New episode wins. zero out existing
                try:
                    await self._log.set_episode_fitness(existing["id"], 0.0)
                except Exception as e:
                    logger.error("ContradictionEngine: failed to zero fitness: %s", e)
                await self._log_contradiction(existing["id"], None, "keep_b", resolution)

            elif resolution.resolution == "merge":
                # Merge. zero out existing, store merged content
                try:
                    await self._log.set_episode_fitness(existing["id"], 0.0)
                except Exception as e:
                    logger.error("ContradictionEngine: failed to zero fitness for merge: %s", e)
                final_content = resolution.merged_content or new_episode_content
                result.final_action = "merged"
                result.final_content = final_content
                await self._log_contradiction(existing["id"], None, "merge", resolution)

            else:  # keep_both
                await self._log_contradiction(existing["id"], None, "keep_both", resolution)

        # If no keep_a was found and we haven't merged, action is "stored"
        if result.final_action not in ("blocked", "merged"):
            result.final_action = "stored"

        return result

    # ------------------------------------------------------------------
    # LLM contradiction check
    # ------------------------------------------------------------------

    async def _check_pair(
        self,
        existing_episode: dict,
        existing_text: str,
        new_text: str,
        client: Any,
    ) -> Optional[ResolutionDetail]:
        """
        Ask Claude whether two episodes contradict each other.

        Returns:
            ResolutionDetail if a real contradiction is found, else None.
        """
        created_at = existing_episode.get("timestamp", "unknown")
        if isinstance(created_at, datetime):
            created_at = created_at.strftime("%Y-%m-%d %H:%M")

        prompt = f"""You are a contradiction detection system for an AI memory store.

Determine if Memory A and Memory B contradict each other.

A contradiction means:
- They cannot both be true at the same time, OR
- One supersedes or invalidates the other, OR
- They represent conflicting preferences/facts about the same topic

Memory A (existing, created {created_at}):
{existing_text}

Memory B (new, just now):
{new_text}

Respond in valid JSON only:
{{
  "is_contradiction": false,
  "confidence": 0.0,
  "contradiction_type": "none",
  "reason": "brief explanation",
  "recommendation": "keep_both",
  "merged_content": ""
}}"""

        try:
            response = await client.messages.create(
                model=config.ANTHROPIC_MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error("ContradictionEngine: invalid JSON from LLM: %s", e)
            return None
        except Exception as e:
            logger.error("ContradictionEngine: LLM call failed: %s", e)
            return None

        if not parsed.get("is_contradiction", False):
            return None

        confidence = float(parsed.get("confidence", 0.0))
        if confidence < config.CONTRADICTION_MIN_CONFIDENCE:
            logger.debug(
                "ContradictionEngine: contradiction detected but confidence %.2f < %.2f. ignoring.",
                confidence,
                config.CONTRADICTION_MIN_CONFIDENCE,
            )
            return None

        recommendation = parsed.get("recommendation", "keep_both")
        merged = parsed.get("merged_content") or None

        logger.info(
            "ContradictionEngine: contradiction found (type=%s, confidence=%.2f, recommendation=%s)",
            parsed.get("contradiction_type", "unknown"),
            confidence,
            recommendation,
        )

        return ResolutionDetail(
            existing_episode_id=existing_episode["id"],
            resolution=recommendation,
            confidence=confidence,
            reason=parsed.get("reason", ""),
            merged_content=merged if recommendation == "merge" else None,
        )

    # ------------------------------------------------------------------
    # DB logging
    # ------------------------------------------------------------------

    async def _log_contradiction(
        self,
        episode_a_id: str,
        episode_b_id: Optional[str],
        resolution: str,
        detail: ResolutionDetail,
    ) -> None:
        """Log a contradiction record. swallows DB errors to avoid crashing chat."""
        try:
            await self._log.log_contradiction(
                episode_a_id=episode_a_id,
                episode_b_id=episode_b_id,
                resolution=resolution,
                reason=detail.reason,
                confidence=detail.confidence,
            )
        except Exception as e:
            logger.error("ContradictionEngine: failed to log contradiction: %s", e)

    # ------------------------------------------------------------------
    # Anthropic client helper
    # ------------------------------------------------------------------

    def _get_client(self) -> Optional[Any]:
        """Return the Anthropic client, lazily creating it if needed."""
        if self._client is not None:
            return self._client
        if not config.ANTHROPIC_API_KEY:
            logger.debug(
                "ContradictionEngine: ANTHROPIC_API_KEY not set. "
                "contradiction checks disabled."
            )
            return None
        try:
            import anthropic
            self._client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
            return self._client
        except ImportError:
            logger.error(
                "ContradictionEngine: 'anthropic' package not installed. "
                "Run: pip install anthropic"
            )
            return None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_engine_instance: Optional[ContradictionEngine] = None


def get_contradiction_engine() -> Optional[ContradictionEngine]:
    """Return the global ContradictionEngine instance (None if not initialised)."""
    return _engine_instance


def set_contradiction_engine(engine: ContradictionEngine) -> None:
    """Register the global ContradictionEngine instance (called from main.py)."""
    global _engine_instance
    _engine_instance = engine
