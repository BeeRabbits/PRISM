"""
MIRROR Oracle: calls Claude API to score PRISM responses.

The oracle evaluates how well a PRISM response matches the user's
preferences and style, using a personalized context file.

Returns a score on the 1-5 scale with reasoning.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

import config
from utils.mirror_context import load_mirror_context

logger = logging.getLogger(__name__)

# Module-level client (lazy init)
_client: Optional[Any] = None


def _get_client() -> Optional[Any]:
    """Return the Anthropic async client, lazily creating if needed."""
    global _client
    if _client is not None:
        return _client
    api_key = config.MIRROR_ANTHROPIC_API_KEY
    if not api_key:
        logger.warning("MIRROR: No API key configured. Oracle scoring disabled.")
        return None
    try:
        import anthropic
        _client = anthropic.AsyncAnthropic(api_key=api_key)
        return _client
    except ImportError:
        logger.error("MIRROR: 'anthropic' package not installed.")
        return None


async def oracle_score(
    user_message: str,
    prism_response: str,
) -> Optional[dict]:
    """
    Call Claude to evaluate a PRISM response.

    Args:
        user_message:   What the user said.
        prism_response: What PRISM replied.

    Returns:
        {"oracle_score": float (1-5), "reasoning": str} or None on failure.
    """
    if not config.MIRROR_ENABLED or config.MIRROR_CONVERGED:
        return None

    client = _get_client()
    if client is None:
        return None

    context = load_mirror_context()

    system_prompt = f"""You are an AI response quality judge for a personalized AI called PRISM.

USER CONTEXT:
{context}

INSTRUCTIONS:
Evaluate the PRISM response based on:
1. Relevance: Does it address what the user actually asked?
2. Personality: Does it match a natural, personalized conversational style?
3. Helpfulness: Is it genuinely useful vs. generic filler?
4. Accuracy: Are any facts or suggestions correct?
5. Conciseness: Is it appropriately brief or detailed for the question?

Score from 1-5:
  1 = Terrible (wrong, generic, unhelpful)
  2 = Poor (partially relevant but mostly filler)
  3 = Acceptable (correct but generic)
  4 = Good (relevant, helpful, some personality)
  5 = Excellent (personalized, insightful, natural)

Respond in JSON only. No markdown fences."""

    user_prompt = f"""Rate this interaction:

USER: {user_message}

PRISM: {prism_response}

{{"oracle_score": <1-5>, "reasoning": "<1-2 sentences>"}}"""

    try:
        response = await client.messages.create(
            model=config.MIRROR_ORACLE_MODEL,
            max_tokens=256,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = response.content[0].text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)

        score = float(parsed.get("oracle_score", 3))
        score = max(1.0, min(5.0, score))

        return {
            "oracle_score": score,
            "reasoning": parsed.get("reasoning", ""),
        }

    except json.JSONDecodeError as e:
        logger.error("MIRROR oracle: invalid JSON response: %s", e)
        return None
    except Exception as e:
        logger.error("MIRROR oracle: API call failed: %s", e)
        return None
