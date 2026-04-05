"""
ExpertRouter: routes input to the appropriate LoRA expert adapter(s).

MoE-LoRA (Mixture of Expert Adapters) approach:
    Instead of one LoRA that tries to learn everything (causing catastrophic
    forgetting), we maintain separate expert adapters that activate selectively.

    Current experts:
      - "style": How PRISM talks — direct, personalized, no generic lists.
                  Trained on high-oracle-score episodes with low MIRROR delta.
      - (future) "reasoning": Deep analytical patterns.
      - (future) "domain": Domain-specific knowledge patterns.

Neuroscience parallel:
    Different brain regions specialize in different functions. The prefrontal
    cortex handles executive function, Broca's area handles language production,
    the hippocampus handles memory. Damage to one doesn't destroy the others.
    MoE-LoRA mirrors this — each expert handles a specific capability.

Architecture:
    The router is lightweight (keyword + heuristic based, zero API cost).
    It returns a list of (expert_name, weight) tuples that the inference
    engine uses to set adapter weights before generation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Expert definitions
# ---------------------------------------------------------------------------

@dataclass
class ExpertConfig:
    """Configuration for a single LoRA expert."""
    name: str
    adapter_path: str
    description: str
    default_weight: float
    # Keywords that boost this expert's activation
    activation_keywords: List[str]


# Expert registry — add new experts here
EXPERTS: Dict[str, ExpertConfig] = {
    "style": ExpertConfig(
        name="style",
        adapter_path="./adapters/experts/style",
        description="Conversational style: direct, personalized, no fluff",
        default_weight=0.4,
        activation_keywords=[],  # Style always active at base weight
    ),
}


def get_available_experts() -> Dict[str, ExpertConfig]:
    """Return experts that have trained adapter files on disk."""
    available = {}
    for name, expert in EXPERTS.items():
        adapter_config = Path(expert.adapter_path) / "adapter_config.json"
        if adapter_config.exists():
            available[name] = expert
    return available


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class ExpertRouter:
    """
    Routes input to expert adapter(s) with appropriate weights.

    The router is intentionally simple — keyword and heuristic based.
    No API calls, no model inference. Just fast pattern matching.

    Usage:
        router = ExpertRouter()
        experts = router.route("Help me plan my workout after work")
        # → [("style", 0.4)]  (style always active)
    """

    def __init__(self) -> None:
        self._experts = get_available_experts()
        if self._experts:
            logger.info(
                "ExpertRouter: %d expert(s) available: %s",
                len(self._experts),
                ", ".join(self._experts.keys()),
            )
        else:
            logger.info("ExpertRouter: no trained experts found.")

    @property
    def has_experts(self) -> bool:
        """Return True if at least one expert adapter is available."""
        return len(self._experts) > 0

    def route(self, user_message: str) -> List[Tuple[str, float]]:
        """
        Determine which experts to activate and their weights.

        Args:
            user_message: The current user message.

        Returns:
            List of (expert_name, weight) tuples.
            Empty list if no experts are available.
        """
        if not self._experts:
            return []

        activations: List[Tuple[str, float]] = []

        for name, expert in self._experts.items():
            weight = expert.default_weight

            # Boost weight if activation keywords match
            if expert.activation_keywords:
                msg_lower = user_message.lower()
                matches = sum(1 for kw in expert.activation_keywords if kw in msg_lower)
                if matches > 0:
                    # Boost by up to 0.3 based on keyword matches
                    boost = min(0.3, matches * 0.1)
                    weight = min(1.0, weight + boost)

            activations.append((name, weight))

        return activations

    def refresh(self) -> None:
        """Re-scan disk for available experts (after training creates new ones)."""
        self._experts = get_available_experts()
        logger.info("ExpertRouter refreshed: %d expert(s) available.", len(self._experts))
