"""
DatasetBuilder. converts logged episodes into a HuggingFace Dataset
formatted for SFTTrainer (Qwen ChatML / instruct format).

Neuroscience-inspired upgrades:
  - Delta-weighted repetition (MIRROR protocol)
  - Spaced repetition replay (Fibonacci intervals)
  - Interleaved training batches (topic round-robin)
  - Mixed knowledge dataset (general + personal to prevent catastrophic forgetting)

Format (one row = one training example):

    <|im_start|>system
    You are PRISM, a persistent intelligent assistant.
    <|im_end|>
    <|im_start|>user
    {user_message}
    <|im_end|>
    <|im_start|>assistant
    {assistant_response}
    <|im_end|>
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

from datasets import Dataset, DatasetDict

import config
from data.experience_log import ExperienceLogger, get_experience_logger
from data.semantic_store import SemanticStore, get_semantic_store

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are PRISM, a persistent intelligent assistant. "
    "You learn from every conversation and grow over time."
)

# Topic keywords for interleaved training. auto-clusters episodes by content
# Topic keywords for interleaved training. Auto-clusters episodes by content.
# These are generic categories. Personal names come from conversation content,
# not from this code, so no personal info is stored here.
_TOPIC_KEYWORDS = {
    "family": ["son", "daughter", "kid", "custody", "parenting", "dad", "mom", "child", "family"],
    "relationship": ["girlfriend", "boyfriend", "partner", "relationship", "love", "together", "marriage"],
    "career": ["work", "job", "career", "promotion", "manager", "salary", "role", "company"],
    "fitness": ["workout", "gym", "exercise", "diet", "weight", "muscle", "health", "fitness"],
    "finance": ["money", "budget", "debt", "income", "save", "invest", "salary", "afford"],
    "media": ["anime", "manga", "movie", "show", "game", "book", "episode", "character", "story"],
    "ai_tech": ["ai", "machine learning", "model", "training", "prism", "neural", "llm", "gpu"],
    "education": ["course", "study", "certification", "degree", "exam", "learn", "school", "university"],
    "productivity": ["routine", "schedule", "morning", "habit", "goal", "discipline", "time"],
    "general": [],  # catch-all for unmatched topics
}


def _format_example(user_message: str, assistant_response: str) -> str:
    """Format a single episode as a ChatML training string."""
    return (
        f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_message}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_response}<|im_end|>"
    )


def _detect_topic(text: str) -> str:
    """Detect the dominant topic of a text using keyword matching."""
    text_lower = text.lower()
    best_topic = "general"
    best_score = 0
    for topic, keywords in _TOPIC_KEYWORDS.items():
        if not keywords:
            continue
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > best_score:
            best_score = score
            best_topic = topic
    return best_topic


def _interleave_by_topic(items: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    """
    Round-robin interleave items by topic so consecutive training examples
    come from different topics. This prevents the model from collapsing
    into one topic's distribution during a training batch.

    Args:
        items: List of (text, episode_id, topic) tuples.

    Returns:
        Reordered list with topics interleaved.
    """
    import random

    buckets: dict[str, list] = defaultdict(list)
    for item in items:
        buckets[item[2]].append(item)

    # Shuffle within each bucket
    for topic_items in buckets.values():
        random.shuffle(topic_items)

    # Round-robin across buckets
    result: List[Tuple[str, str, str]] = []
    topic_keys = list(buckets.keys())
    random.shuffle(topic_keys)

    while any(buckets[k] for k in topic_keys):
        for key in topic_keys:
            if buckets[key]:
                result.append(buckets[key].pop(0))

    return result


class DatasetBuilder:
    """
    Builds a HuggingFace DatasetDict from high-fitness episodes and confirmed
    semantic memories, with neuroscience-inspired training optimizations.

    Features:
      - Delta-weighted repetition (MIRROR protocol)
      - Spaced repetition replay episodes
      - Interleaved topic ordering
      - Mixed general knowledge to prevent catastrophic forgetting

    Args:
        experience_logger: The ExperienceLogger instance to query.
        semantic_store:    SemanticStore for confirmed semantic memories.
        min_fitness:       Minimum fitness score to include.
        min_episodes:      Minimum number of episodes required to proceed.
    """

    def __init__(
        self,
        experience_logger: Optional[ExperienceLogger] = None,
        semantic_store: Optional[SemanticStore] = None,
        min_fitness: Optional[float] = None,
        min_episodes: Optional[int] = None,
    ) -> None:
        self._log = experience_logger or get_experience_logger()
        self._store = semantic_store or get_semantic_store()
        self._min_fitness = min_fitness if min_fitness is not None else config.TRAINING_MIN_FITNESS
        self._min_episodes = min_episodes if min_episodes is not None else config.TRAINING_MIN_EPISODES
        self._last_semantic_ids: List[str] = []
        self._last_avg_delta: Optional[float] = None
        self._last_replay_ids: List[str] = []

    async def build(self) -> Optional[DatasetDict]:
        """
        Fetch episodes + confirmed semantics + replay episodes + general knowledge
        and build a train/eval DatasetDict with interleaved topic ordering.

        Returns:
            DatasetDict with "train" and "eval" splits,
            or None if not enough episode data.
        """
        import random

        episodes = await self._log.get_episodes_with_delta(
            min_score=self._min_fitness,
            limit=2_000,
        )

        n = len(episodes)
        if n < self._min_episodes:
            logger.warning(
                "Not enough data yet: %d episodes available, need %d. Skipping build.",
                n,
                self._min_episodes,
            )
            return None

        logger.info("Building dataset from %d episodes (min_fitness=%.2f).", n, self._min_fitness)

        # -----------------------------------------------------------
        # Phase 1: Delta-weighted personal episodes
        # -----------------------------------------------------------
        items: List[Tuple[str, str, str]] = []  # (text, id, topic)
        delta_sum = 0.0
        delta_count = 0
        for ep in episodes:
            text = _format_example(ep["user_message"], ep["assistant_response"])
            topic = _detect_topic(f"{ep['user_message']} {ep['assistant_response']}")
            delta = ep.get("mirror_delta")
            if delta is not None and delta > 0:
                weight = 1 + min(2.0, delta)
                reps = max(1, round(weight))
                delta_sum += delta
                delta_count += 1
            else:
                reps = 1
            for _ in range(reps):
                items.append((text, ep["id"], topic))

        self._last_avg_delta = (delta_sum / delta_count) if delta_count > 0 else None
        if self._last_avg_delta is not None:
            logger.info("MIRROR: avg delta for training batch = %.3f (%d episodes with delta)",
                        self._last_avg_delta, delta_count)

        # -----------------------------------------------------------
        # Enforce 60/30/10 split:
        #   60% personal episodes (items so far)
        #   30% general knowledge
        #   10% spaced replay
        # -----------------------------------------------------------
        personal_items = list(items)  # snapshot personal episodes
        personal_target = len(personal_items)

        # Calculate target counts from the 60/30/10 ratio
        # personal_target = 60% of total → total = personal_target / 0.60
        total_target = int(personal_target / 0.60) if personal_target > 0 else 0
        replay_target = int(total_target * config.SPACED_REPLAY_RATIO)
        general_target = int(total_target * config.MIXED_KNOWLEDGE_RATIO)

        # -----------------------------------------------------------
        # Phase 2: Spaced Repetition replay episodes (10%)
        # -----------------------------------------------------------
        replay_items: List[Tuple[str, str, str]] = []
        if config.SPACED_REPLAY_ENABLED:
            replay_episodes = await self._log.get_due_episodes(
                limit=max(replay_target, config.SPACED_REPLAY_MAX_PER_BATCH),
            )
            self._last_replay_ids = [ep["id"] for ep in replay_episodes]
            for ep in replay_episodes[:replay_target]:
                text = _format_example(ep["user_message"], ep["assistant_response"])
                topic = _detect_topic(f"{ep['user_message']} {ep['assistant_response']}")
                replay_items.append((text, f"replay:{ep['id']}", topic))
            if replay_items:
                logger.info(
                    "Spaced Replay: %d episodes (target %d = %.0f%% of batch).",
                    len(replay_items), replay_target,
                    100 * len(replay_items) / max(1, total_target),
                )
        items.extend(replay_items)

        # -----------------------------------------------------------
        # Phase 3: Confirmed semantic memories (part of personal 60%)
        # -----------------------------------------------------------
        semantics = await self._store.get_training_ready_semantics()
        self._last_semantic_ids = [s["id"] for s in semantics]
        if semantics:
            logger.info("Appending %d confirmed semantic memories to training data.", len(semantics))
            for sem in semantics:
                items.append((
                    self._store.format_for_training(sem),
                    f"semantic:{sem['id']}",
                    _detect_topic(sem["content"]),
                ))

        # -----------------------------------------------------------
        # Phase 4: Mixed Knowledge. general Q&A (30%)
        # -----------------------------------------------------------
        if config.MIXED_KNOWLEDGE_ENABLED:
            general_items = self._load_general_knowledge()
            if general_items:
                random.shuffle(general_items)
                general_sample = general_items[:general_target]
                actual_pct = 100 * len(general_sample) / max(1, len(items) + len(general_sample))
                logger.info(
                    "Mixed Knowledge: %d general examples (target %d = %.0f%% of batch).",
                    len(general_sample), general_target, actual_pct,
                )
                items.extend(general_sample)

        logger.info(
            "Dataset split: %d personal (60%%) + %d replay (10%%) + %d general (30%%) = %d total.",
            len(personal_items), len(replay_items),
            len(items) - len(personal_items) - len(replay_items),
            len(items),
        )

        # -----------------------------------------------------------
        # Phase 5: Interleaved ordering (or plain shuffle)
        # -----------------------------------------------------------
        if config.INTERLEAVE_ENABLED and len(items) > 10:
            items = _interleave_by_topic(items)
            logger.info("Interleaved training: topics round-robin ordered.")
        else:
            random.shuffle(items)

        texts = [t[0] for t in items]
        ids = [t[1] for t in items]

        raw_dataset = Dataset.from_dict({"text": texts, "episode_id": ids})

        # 90/10 train/eval split
        split = raw_dataset.train_test_split(test_size=0.1, seed=42)
        dataset = DatasetDict({"train": split["train"], "eval": split["test"]})

        logger.info(
            "Dataset built: %d train, %d eval examples "
            "(%d semantic, %d replay, %d general).",
            len(dataset["train"]),
            len(dataset["eval"]),
            len(semantics),
            len(self._last_replay_ids),
            len([i for i in ids if i.startswith("general:")]),
        )
        return dataset

    async def advance_replay_after_training(self) -> None:
        """
        After a successful training run, advance the replay schedule
        for all spaced-repetition episodes that were included.
        """
        if self._last_replay_ids:
            await self._log.advance_replay_schedule(self._last_replay_ids)
            self._last_replay_ids = []

    async def init_replay_for_new_episodes(self, episode_ids: List[str]) -> None:
        """
        Initialize spaced replay schedules for newly trained episodes.
        Called after mark_training_used().
        """
        if config.SPACED_REPLAY_ENABLED and episode_ids:
            await self._log.init_replay_schedule(episode_ids)

    async def mark_semantics_used(self) -> None:
        """Mark the semantic memories included in the last build() call as training-used."""
        if self._last_semantic_ids:
            await self._store.mark_training_used(self._last_semantic_ids)
            self._last_semantic_ids = []

    async def get_episode_ids_for_training(self) -> List[str]:
        """Return IDs of episodes that would be selected for the next training run."""
        episodes = await self._log.get_high_fitness_episodes(
            min_score=self._min_fitness,
            limit=2_000,
        )
        return [ep["id"] for ep in episodes]

    # ------------------------------------------------------------------
    # Mixed Knowledge loader
    # ------------------------------------------------------------------

    @staticmethod
    def _load_general_knowledge() -> List[Tuple[str, str, str]]:
        """
        Load general knowledge examples from data/general/*.jsonl.

        Each JSONL line should have:
          {"instruction": "...", "output": "..."}
          OR
          {"user_message": "...", "assistant_response": "..."}

        Returns:
            List of (text, id, topic) tuples.
        """
        general_path = Path(config.MIXED_KNOWLEDGE_PATH)
        if not general_path.exists():
            logger.info("Mixed Knowledge: %s not found. skipping.", general_path)
            return []

        items: List[Tuple[str, str, str]] = []
        for jsonl_file in sorted(general_path.glob("*.jsonl")):
            try:
                for line_num, line in enumerate(jsonl_file.read_text().splitlines()):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    user = obj.get("user_message") or obj.get("instruction") or obj.get("input", "")
                    assistant = obj.get("assistant_response") or obj.get("output") or obj.get("response", "")
                    if not user or not assistant:
                        continue

                    text = _format_example(user, assistant)
                    topic = _detect_topic(f"{user} {assistant}")
                    items.append((text, f"general:{jsonl_file.stem}:{line_num}", topic))
            except Exception as e:
                logger.warning("Mixed Knowledge: error reading %s: %s", jsonl_file, e)

        if items:
            logger.info("Mixed Knowledge: loaded %d general examples from %s.", len(items), general_path)
        return items
