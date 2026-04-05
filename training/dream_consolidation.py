"""
DreamConsolidation. nightly pre-training pipeline that compresses clusters
of high-fitness episodic memories into validated semantic memories.

Pipeline (run nightly before LoRA training):
  1. Fetch consolidation-eligible episodes (high-fitness, old, unused).
  2. Cluster by text similarity (greedy cosine, trigram vectors).
  3. For each cluster ≥ min_cluster_size, call Claude to consolidate.
  4. Store high-confidence results as provisional semantic memories.
  5. Validate existing provisional semantics against recent episodes.
  6. Promote semantics with validation_count ≥ threshold to confirmed.
  7. Prune provisional semantics older than SEMANTIC_PRUNE_DAYS with
     zero validations (likely hallucinated).

The LLM (Claude) does the semantic understanding. Text similarity is only
used for pre-filtering to keep the pipeline fast.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import config
from data.experience_log import ExperienceLogger, get_experience_logger
from data.semantic_store import SemanticStore, get_semantic_store
from utils.anthropic_client import get_async_anthropic_client
from utils.text_similarity import cosine_similarity_trigram

import torch

logger = logging.getLogger(__name__)

# Last consolidation report. stored for the /admin/consolidation-report endpoint
_last_report: Optional[dict] = None


# ---------------------------------------------------------------------------
# MIRROR self-scoring (uses local PRISM model, not external API)
# ---------------------------------------------------------------------------

async def self_score(
    model,
    tokenizer,
    user_message: str,
    prism_response: str,
) -> Optional[dict]:
    """
    Ask PRISM to evaluate the quality of its own response on a 1-5 scale.

    Uses the local model to generate a self-assessment. This is the
    self-referential component of the MIRROR protocol.

    Args:
        model:          The loaded causal LM.
        tokenizer:      The associated tokenizer.
        user_message:   What the user asked.
        prism_response: What PRISM replied.

    Returns:
        {"self_score": float (1-5), "confidence": float (0-1)} or None.
    """
    prompt = (
        "<|im_start|>system\n"
        "You are a response quality evaluator. Rate the assistant response "
        "on a scale of 1-5 where 1=terrible and 5=excellent. "
        "Consider relevance, helpfulness, and naturalness. "
        "Reply with ONLY a JSON object: {\"score\": N, \"confidence\": 0.X}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Rate this response:\n\n"
        f"User asked: {user_message[:500]}\n\n"
        f"Assistant replied: {prism_response[:500]}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    try:
        device = next(model.parameters()).device
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = enc["input_ids"].to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=64,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output[0][input_ids.shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Try JSON parse first
        try:
            import json as _json
            text_clean = re.sub(r"^```(?:json)?\s*", "", text)
            text_clean = re.sub(r"\s*```$", "", text_clean)
            parsed = _json.loads(text_clean)
            score = float(parsed.get("score", 3))
            confidence = float(parsed.get("confidence", 0.5))
        except (ValueError, _json.JSONDecodeError):
            # Fallback: extract first number from output
            numbers = re.findall(r"(\d(?:\.\d+)?)", text)
            if numbers:
                score = float(numbers[0])
                confidence = 0.3  # low confidence on fallback parse
            else:
                return None

        score = max(1.0, min(5.0, score))
        confidence = max(0.0, min(1.0, confidence))
        return {"self_score": score, "confidence": confidence}

    except Exception as e:
        logger.error("MIRROR self-score failed: %s", e)
        return None


def get_last_consolidation_report() -> Optional[dict]:
    """Return the most recent consolidation report dict."""
    return _last_report


class DreamConsolidation:
    """
    Compresses episodic memory clusters into validated semantic memories.

    Args:
        experience_logger: Source of episodic episodes.
        semantic_store:    Destination for semantic memories.
        anthropic_client:  An instantiated anthropic.AsyncAnthropic client,
                           or None to attempt lazy creation from config.
    """

    def __init__(
        self,
        experience_logger: Optional[ExperienceLogger] = None,
        semantic_store: Optional[SemanticStore] = None,
        anthropic_client: Optional[Any] = None,
    ) -> None:
        self._log = experience_logger or get_experience_logger()
        self._store = semantic_store or get_semantic_store()
        self._client = anthropic_client  # may be None; lazily resolved

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        min_fitness: Optional[float] = None,
        min_cluster_size: Optional[int] = None,
    ) -> dict:
        """
        Run the full dream consolidation pipeline with TMR (Targeted Memory
        Reactivation) when enabled.

        TMR phase (neuroscience-inspired):
          - Prioritize high-delta (salient) episodes for consolidation first.
          - After standard consolidation, reactivate salient memories by
            pairing them with related semantic memories for reinforced encoding.

        Args:
            min_fitness:      Override CONSOLIDATION_MIN_FITNESS.
            min_cluster_size: Override CONSOLIDATION_MIN_CLUSTER_SIZE.

        Returns:
            Report dict with counts and duration.
        """
        global _last_report

        min_fit = min_fitness if min_fitness is not None else config.CONSOLIDATION_MIN_FITNESS
        min_size = min_cluster_size if min_cluster_size is not None else config.CONSOLIDATION_MIN_CLUSTER_SIZE

        report: dict = {
            "clusters_found": 0,
            "clusters_skipped_low_confidence": 0,
            "clusters_skipped_too_varied": 0,
            "semantics_created": 0,
            "semantics_promoted": 0,
            "semantics_pruned": 0,
            "tmr_salient_count": 0,
            "tmr_reactivations": 0,
            "duration_seconds": 0.0,
            "ran_at": datetime.utcnow().isoformat(),
        }
        t0 = time.monotonic()

        # Step 1. Fetch candidates
        candidates = await self._log.get_consolidation_candidates(
            min_fitness=min_fit,
            hours_old=48,
            limit=500,
        )
        logger.info("DreamConsolidation: %d consolidation candidates fetched.", len(candidates))

        if not candidates:
            logger.info("DreamConsolidation: no candidates. skipping.")
            report["duration_seconds"] = round(time.monotonic() - t0, 2)
            _last_report = report
            return report

        # TMR Step. Sort by salience (mirror_delta) so high-delta episodes
        # are clustered and consolidated first. In neuroscience, emotionally
        # salient memories get preferential consolidation during sleep.
        if config.TMR_ENABLED:
            candidates = self._sort_by_salience(candidates)
            salient = [
                c for c in candidates
                if (c.get("mirror_delta") or 0) >= config.TMR_SALIENT_DELTA_THRESHOLD
            ]
            report["tmr_salient_count"] = len(salient)
            if salient:
                logger.info(
                    "TMR: %d salient episodes (delta ≥ %.1f) prioritized for consolidation.",
                    len(salient), config.TMR_SALIENT_DELTA_THRESHOLD,
                )

        # Step 2. Cluster
        clusters = self._cluster_episodes(
            candidates,
            threshold=config.CONSOLIDATION_SIMILARITY_THRESHOLD,
            min_size=min_size,
        )
        report["clusters_found"] = len(clusters)
        logger.info("DreamConsolidation: %d valid clusters (≥%d episodes).", len(clusters), min_size)

        # Step 3-4. Consolidate each cluster
        client = self._get_client()
        all_consolidated_episode_ids: List[str] = []

        for cluster in clusters:
            result = await self._consolidate_cluster(cluster, client)
            if result is None:
                report["clusters_skipped_too_varied"] += 1
                continue
            if result.get("skipped_low_confidence"):
                report["clusters_skipped_low_confidence"] += 1
                continue

            sem_id = await self._store.create_semantic_memory(
                content=result["semantic_memory"],
                source_episode_ids=[ep["id"] for ep in cluster],
                key_pattern=result["key_pattern"],
                confidence=result["confidence"],
            )
            all_consolidated_episode_ids.extend([ep["id"] for ep in cluster])
            report["semantics_created"] += 1
            logger.info(
                "DreamConsolidation: consolidated %d episodes → '%s'",
                len(cluster),
                result["key_pattern"],
            )

            # Extract knowledge triples into the hippocampal index
            triples = result.get("triples", [])
            if triples:
                from data.knowledge_graph import get_knowledge_graph
                kg = get_knowledge_graph()
                source_ids = [ep["id"] for ep in cluster] + [sem_id]
                added = await kg.add_triples_batch(
                    triples,
                    source_type="consolidation",
                    source_ids=source_ids,
                )
                report["triples_extracted"] = report.get("triples_extracted", 0) + len(added)
                logger.info(
                    "KnowledgeGraph: extracted %d triples from cluster '%s'",
                    len(added), result["key_pattern"],
                )

        # Mark source episodes as used
        if all_consolidated_episode_ids:
            await self._log.mark_consolidation_used(all_consolidated_episode_ids)

        # TMR Reactivation. pair salient unclustered episodes with existing
        # semantic memories for targeted memory reinforcement
        if config.TMR_ENABLED:
            reactivations = await self._tmr_reactivation(candidates, client)
            report["tmr_reactivations"] = reactivations

        # Step 5-7. Validate and promote/prune provisional semantics
        promoted, pruned = await self._validate_provisional_semantics(client)
        report["semantics_promoted"] = promoted
        report["semantics_pruned"] = pruned

        report["duration_seconds"] = round(time.monotonic() - t0, 2)
        _last_report = report

        logger.info(
            "DreamConsolidation complete: %d created, %d promoted, %d pruned, "
            "%d TMR reactivations. (%.1fs)",
            report["semantics_created"],
            report["semantics_promoted"],
            report["semantics_pruned"],
            report["tmr_reactivations"],
            report["duration_seconds"],
        )
        return report

    # ------------------------------------------------------------------
    # TMR (Targeted Memory Reactivation)
    # ------------------------------------------------------------------

    @staticmethod
    def _sort_by_salience(candidates: List[dict]) -> List[dict]:
        """
        Sort candidates by salience (mirror_delta descending).
        High-delta episodes are emotionally salient. they represent
        the biggest gap between oracle and self-assessment.
        """
        def _salience(ep: dict) -> float:
            delta = ep.get("mirror_delta") or 0.0
            fitness = ep.get("fitness_score", 1.0)
            return delta * fitness  # combined salience signal
        return sorted(candidates, key=_salience, reverse=True)

    async def _tmr_reactivation(
        self,
        candidates: List[dict],
        client: Optional[Any],
    ) -> int:
        """
        TMR Reactivation phase: for each salient episode not yet consolidated,
        find the closest existing semantic memory and create a reinforcement
        pairing. This strengthens the connection between episodic and semantic
        representations.

        Returns:
            Number of reactivation pairings created.
        """
        salient = [
            c for c in candidates
            if (c.get("mirror_delta") or 0) >= config.TMR_SALIENT_DELTA_THRESHOLD
            and not c.get("used_in_consolidation", False)
        ]

        if not salient:
            return 0

        # Get existing confirmed semantics to pair with
        confirmed = await self._store.get_confirmed_semantics(limit=100)
        if not confirmed:
            logger.info("TMR: no confirmed semantics for reactivation pairing.")
            return 0

        reactivation_count = 0
        max_reactivations = int(len(candidates) * config.TMR_REACTIVATION_RATIO)

        for ep in salient[:max_reactivations]:
            ep_text = f"{ep['user_message']} {ep['assistant_response']}"

            # Find most similar semantic memory
            best_sim = 0.0
            best_sem = None
            for sem in confirmed:
                sim = cosine_similarity_trigram(ep_text, sem["content"])
                if sim > best_sim:
                    best_sim = sim
                    best_sem = sem

            if best_sem and best_sim > 0.3:
                # Create a TMR-reinforced semantic: the salient episode
                # strengthens the related semantic memory
                await self._store.increment_validation(best_sem["id"])
                reactivation_count += 1
                logger.debug(
                    "TMR reactivation: episode %s (delta=%.1f) → semantic '%s' (sim=%.2f)",
                    ep["id"],
                    ep.get("mirror_delta", 0),
                    best_sem["key_pattern"],
                    best_sim,
                )

        if reactivation_count:
            logger.info("TMR: %d salient episodes reactivated with semantic pairings.", reactivation_count)

        return reactivation_count

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def _cluster_episodes(
        self,
        episodes: List[dict],
        threshold: float,
        min_size: int,
    ) -> List[List[dict]]:
        """
        Greedy cosine-similarity clustering.

        Each episode's text is a concatenation of user_message + assistant_response.
        Clusters with fewer than min_size members are discarded.
        """
        texts = [f"{ep['user_message']} {ep['assistant_response']}" for ep in episodes]
        remaining = list(range(len(episodes)))
        clusters: List[List[dict]] = []

        while remaining:
            seed_idx = remaining.pop(0)
            cluster = [episodes[seed_idx]]
            still_remaining = []

            for idx in remaining:
                sim = cosine_similarity_trigram(texts[seed_idx], texts[idx])
                if sim >= threshold:
                    cluster.append(episodes[idx])
                else:
                    still_remaining.append(idx)

            remaining = still_remaining

            if len(cluster) >= min_size:
                clusters.append(cluster)

        return clusters

    # ------------------------------------------------------------------
    # LLM consolidation
    # ------------------------------------------------------------------

    async def _consolidate_cluster(
        self,
        cluster: List[dict],
        client: Optional[Any],
    ) -> Optional[dict]:
        """
        Ask Claude to consolidate a cluster into one semantic memory.

        Returns:
            dict with keys: semantic_memory, key_pattern, confidence
            dict with skipped_low_confidence=True if confidence < threshold
            None if can_consolidate=False or LLM call failed
        """
        if client is None:
            logger.warning("DreamConsolidation: no Anthropic client. skipping cluster consolidation.")
            return None

        formatted = "\n\n".join(
            f"Interaction {i+1}:\nUser: {ep['user_message']}\nAssistant: {ep['assistant_response']}"
            for i, ep in enumerate(cluster[:10])  # cap at 10 to keep prompt size reasonable
        )

        prompt = f"""You are a memory consolidation system for an AI assistant.

Below are several related episodic interactions from the same user.
Your task is to identify the underlying pattern, preference, or fact that \
these interactions share and compress them into ONE precise semantic memory.

Rules:
- Only state what is clearly supported by MULTIPLE interactions
- Do NOT extrapolate or invent. be conservative
- If the interactions are too varied to compress meaningfully, return can_consolidate: false
- Maximum 3 sentences for the semantic memory

Interactions:
{formatted}

Also extract specific factual knowledge as (subject, predicate, object) triples.
Focus on: personal facts, relationships, preferences, habits, goals, dates.
Use the person's actual name as subject, not "the user".

Respond in valid JSON only:
{{
  "can_consolidate": true,
  "semantic_memory": "the compressed memory text",
  "key_pattern": "one-line summary of the pattern found",
  "confidence": 0.0,
  "reasoning": "brief explanation of what pattern was found",
  "triples": [
    {{"subject": "Brandon", "predicate": "works at", "object": "AWS"}},
    {{"subject": "Brandon", "predicate": "has son named", "object": "Landen"}}
  ]
}}"""

        try:
            response = await client.messages.create(
                model=config.ANTHROPIC_MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            # Strip markdown code fences if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error("DreamConsolidation: invalid JSON from LLM: %s", e)
            return None
        except Exception as e:
            logger.error("DreamConsolidation: LLM call failed: %s", e)
            return None

        if not parsed.get("can_consolidate", False):
            logger.debug(
                "DreamConsolidation: cluster not consolidatable. %s",
                parsed.get("reasoning", ""),
            )
            return None

        confidence = float(parsed.get("confidence", 0.0))
        if confidence < config.CONSOLIDATION_MIN_CONFIDENCE:
            logger.debug(
                "DreamConsolidation: confidence %.2f < %.2f. skipping.",
                confidence,
                config.CONSOLIDATION_MIN_CONFIDENCE,
            )
            return {"skipped_low_confidence": True}

        return {
            "semantic_memory": parsed["semantic_memory"],
            "key_pattern": parsed["key_pattern"],
            "confidence": confidence,
            "triples": parsed.get("triples", []),
        }

    # ------------------------------------------------------------------
    # Validation and promotion
    # ------------------------------------------------------------------

    async def _validate_provisional_semantics(
        self,
        client: Optional[Any],
    ) -> Tuple[int, int]:
        """
        Validate provisional semantics against recent episodes.

        For each provisional semantic:
          - Check similarity against last-7-days episodes.
          - Increment validation_count for each matching episode.
          - Promote when validation_count >= SEMANTIC_VALIDATION_THRESHOLD.
          - Prune if older than SEMANTIC_PRUNE_DAYS with zero validations.

        Returns:
            (promoted_count, pruned_count)
        """
        provisionals = await self._store.get_provisional_semantics()
        if not provisionals:
            return 0, 0

        recent_episodes = await self._log.get_recent_episodes(days=7, min_score=0.5, limit=200)
        prune_cutoff = datetime.utcnow() - timedelta(days=config.SEMANTIC_PRUNE_DAYS)

        promoted = 0
        pruned = 0

        for sem in provisionals:
            created_at: datetime = sem["created_at"]
            current_count: int = sem["validation_count"]

            # Check if it should be pruned (too old, never validated)
            if created_at < prune_cutoff and current_count < 1:
                await self._store.delete_semantic(sem["id"])
                pruned += 1
                logger.info("DreamConsolidation: pruned unvalidated semantic %s", sem["id"])
                continue

            # Validate against recent episodes
            new_matches = 0
            for ep in recent_episodes:
                ep_text = f"{ep['user_message']} {ep['assistant_response']}"
                sim = cosine_similarity_trigram(sem["content"], ep_text)
                if sim >= config.CONSOLIDATION_SIMILARITY_THRESHOLD:
                    new_matches += 1

            if new_matches > 0:
                for _ in range(new_matches):
                    current_count = await self._store.increment_validation(sem["id"])

            if current_count >= config.SEMANTIC_VALIDATION_THRESHOLD:
                await self._store.promote_to_confirmed(sem["id"])
                promoted += 1
                logger.info(
                    "DreamConsolidation: promoted semantic %s (validated %d times)",
                    sem["id"],
                    current_count,
                )

        return promoted, pruned

    # ------------------------------------------------------------------
    # Anthropic client helper
    # ------------------------------------------------------------------

    def _get_client(self) -> Optional[Any]:
        """Return the Anthropic client, lazily creating it if needed."""
        if self._client is not None:
            return self._client
        self._client = get_async_anthropic_client()
        return self._client


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_consolidation_instance: Optional[DreamConsolidation] = None


def get_dream_consolidation() -> Optional[DreamConsolidation]:
    """Return the global DreamConsolidation instance (None if not initialised)."""
    return _consolidation_instance


def set_dream_consolidation(instance: DreamConsolidation) -> None:
    """Register the global DreamConsolidation instance (called from main.py)."""
    global _consolidation_instance
    _consolidation_instance = instance
