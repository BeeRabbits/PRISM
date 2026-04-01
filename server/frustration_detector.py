"""
Frustration Detector: regex-based sentiment analysis before LLM inference.

Inspired by Claude Code's 21-pattern frustration detection system.
Zero API calls. Detects user frustration from raw message text and
provides system prompt modifiers to adjust PRISM's response style.

Detected sentiment is stored as episode metadata for training signal analysis.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List

logger = logging.getLogger(__name__)


class FrustrationTier(IntEnum):
    """Frustration severity levels."""
    NONE = 0
    MILD = 1
    FRUSTRATED = 2
    ANGRY = 3


@dataclass
class FrustrationResult:
    """Result of frustration detection analysis."""
    tier: FrustrationTier
    matched_patterns: List[str] = field(default_factory=list)
    recommended_modifier: str = ""


# Pattern definitions: (name, compiled_regex, tier_value)
# Higher tier_value patterns contribute more to the final score.
_PATTERNS: List[tuple[str, re.Pattern, int]] = [
    # Tier 3: Explicit anger / profanity
    ("profanity_strong", re.compile(r"\b(fuck|fucking|fucked|shit|bullshit|ass|asshole|dumbass|goddamn)\b", re.IGNORECASE), 3),
    ("direct_insult", re.compile(r"\b(you('re| are) (stupid|useless|terrible|awful|garbage|trash|worthless|incompetent|dumb))\b", re.IGNORECASE), 3),
    ("rage_punctuation", re.compile(r"[!?]{4,}"), 3),

    # Tier 2: Frustration / exasperation
    ("profanity_mild", re.compile(r"\b(damn|crap|hell|wtf|omfg|ffs|smh)\b", re.IGNORECASE), 2),
    ("repetition_marker", re.compile(r"\b(i (already|just) (told|said|asked|mentioned))\b", re.IGNORECASE), 2),
    ("again_emphasis", re.compile(r"\b(again|once more|for the .+ time|how many times)\b", re.IGNORECASE), 2),
    ("broken_claim", re.compile(r"\b(this is broken|doesn't work|isn't working|still broken|not working)\b", re.IGNORECASE), 2),
    ("caps_shouting", re.compile(r"\b[A-Z]{5,}\b"), 2),
    ("exasperation", re.compile(r"\b(seriously|come on|are you kidding|give me a break|unbelievable)\b", re.IGNORECASE), 2),
    ("wrong_claim", re.compile(r"\b(that('s| is) (wrong|incorrect|not right|not what i))\b", re.IGNORECASE), 2),

    # Tier 1: Mild annoyance / impatience
    ("dismissal", re.compile(r"\b(whatever|forget it|never mind|nevermind|nvm|just stop)\b", re.IGNORECASE), 1),
    ("sarcasm", re.compile(r"\b(oh great|thanks for nothing|wow really|oh wonderful|gee thanks|oh perfect)\b", re.IGNORECASE), 1),
    ("ellipsis_trail", re.compile(r"\.{4,}"), 1),
    ("impatience", re.compile(r"\b(hurry|just do it|stop (explaining|talking)|get to the point)\b", re.IGNORECASE), 1),
    ("confusion_blame", re.compile(r"\b(you (lost|confused|forgot|missed|ignored))\b", re.IGNORECASE), 1),
    ("no_not_that", re.compile(r"\b(no[,.]? not that|that's not what i|i didn't (ask|say|mean))\b", re.IGNORECASE), 1),
    ("disappointment", re.compile(r"\b(disappointing|let me down|expected better|not helpful|unhelpful)\b", re.IGNORECASE), 1),
]

# System prompt modifiers per tier
_MODIFIERS = {
    FrustrationTier.NONE: "",
    FrustrationTier.MILD: (
        "The user may be slightly impatient. Be concise and direct. "
        "Avoid unnecessary preamble."
    ),
    FrustrationTier.FRUSTRATED: (
        "The user seems frustrated. Acknowledge their frustration briefly, "
        "then provide a direct answer. Skip pleasantries and get to the point."
    ),
    FrustrationTier.ANGRY: (
        "The user is clearly upset. Apologize briefly for the difficulty, "
        "be extremely direct, and simplify your response. "
        "Do not add qualifiers or hedge."
    ),
}


def detect_frustration(message: str) -> FrustrationResult:
    """
    Analyze a user message for frustration signals.

    Returns a FrustrationResult with the detected tier, matched pattern
    names, and a recommended system prompt modifier.
    """
    matched: List[str] = []
    score = 0

    for name, pattern, tier_value in _PATTERNS:
        if pattern.search(message):
            matched.append(name)
            score += tier_value

    # Map cumulative score to tier
    if score >= 5:
        tier = FrustrationTier.ANGRY
    elif score >= 3:
        tier = FrustrationTier.FRUSTRATED
    elif score >= 1:
        tier = FrustrationTier.MILD
    else:
        tier = FrustrationTier.NONE

    result = FrustrationResult(
        tier=tier,
        matched_patterns=matched,
        recommended_modifier=_MODIFIERS[tier],
    )

    if tier > FrustrationTier.NONE:
        logger.info(
            "Frustration detected: tier=%s score=%d patterns=%s",
            tier.name, score, matched,
        )

    return result
