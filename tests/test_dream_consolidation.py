"""
Tests for DreamConsolidation pipeline.

Uses real (temp) SQLite databases but mocks the Anthropic LLM client.
No GPU required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from data.experience_log import ExperienceLogger
from data.semantic_store import SemanticStore
from training.dream_consolidation import DreamConsolidation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def logger_with_old_episodes(tmp_path):
    """Logger with 10 old high-fitness episodic episodes ready for consolidation."""
    db_url = f"sqlite+aiosqlite:///{tmp_path}/dream_test.db"
    log = ExperienceLogger(db_url=db_url)
    await log.init()
    old_ts = datetime.utcnow() - timedelta(hours=72)
    for i in range(10):
        eid = await log.log_episode(
            user_message=f"How do I write Python code efficiently?",
            assistant_response=f"Use list comprehensions and avoid loops where possible.",
            full_prompt=f"full {i}",
            session_id="test_user",
        )
        await log.update_fitness(eid, score=0.85, signal_type="explicit")
        # Manually push timestamp back to simulate old episodes
        from sqlalchemy import update, text
        async with log._session_factory() as session:
            from data.experience_log import EpisodeRow
            await session.execute(
                update(EpisodeRow)
                .where(EpisodeRow.id == eid)
                .values(timestamp=old_ts)
            )
            await session.commit()
    return log


@pytest.fixture
async def semantic_store(tmp_path):
    db_url = f"sqlite+aiosqlite:///{tmp_path}/dream_test.db"
    store = SemanticStore(db_url=db_url)
    return store


def _make_mock_client(can_consolidate=True, confidence=0.9, semantic="User prefers concise Python."):
    """Build a mock Anthropic client that returns a canned consolidation response."""
    client = MagicMock()
    response_obj = MagicMock()
    response_obj.content = [MagicMock()]
    response_obj.content[0].text = json.dumps({
        "can_consolidate": can_consolidate,
        "semantic_memory": semantic,
        "key_pattern": "Prefers concise Python code",
        "confidence": confidence,
        "reasoning": "Multiple interactions about Python efficiency",
    })
    client.messages = MagicMock()
    client.messages.create = AsyncMock(return_value=response_obj)
    return client


# ---------------------------------------------------------------------------
# Cluster formation
# ---------------------------------------------------------------------------

class TestClusterFormation:
    def test_similar_episodes_cluster_together(self):
        consolidation = DreamConsolidation()
        episodes = [
            {"id": "1", "user_message": "Python list comprehensions are great",
             "assistant_response": "Yes they are fast and readable"},
            {"id": "2", "user_message": "Python list comprehension best practices",
             "assistant_response": "Use them for simple transformations"},
            {"id": "3", "user_message": "Tell me about quantum physics",
             "assistant_response": "Quantum physics deals with subatomic particles"},
        ]
        clusters = consolidation._cluster_episodes(episodes, threshold=0.3, min_size=2)
        # The two Python episodes should cluster together
        assert len(clusters) == 1
        assert len(clusters[0]) == 2
        ids_in_cluster = {ep["id"] for ep in clusters[0]}
        assert "1" in ids_in_cluster
        assert "2" in ids_in_cluster

    def test_min_size_filters_small_clusters(self):
        consolidation = DreamConsolidation()
        episodes = [
            {"id": "1", "user_message": "hello world", "assistant_response": "hi"},
            {"id": "2", "user_message": "completely different topic", "assistant_response": "ok"},
        ]
        clusters = consolidation._cluster_episodes(episodes, threshold=0.9, min_size=3)
        assert clusters == []

    def test_cosine_similarity_identical_texts(self):
        from utils.text_similarity import cosine_similarity_trigram
        sim = cosine_similarity_trigram("hello world foo", "hello world foo")
        assert sim == pytest.approx(1.0, abs=0.01)

    def test_cosine_similarity_different_texts(self):
        from utils.text_similarity import cosine_similarity_trigram
        sim = cosine_similarity_trigram("python programming", "quantum physics")
        assert sim < 0.3


# ---------------------------------------------------------------------------
# Consolidation prompt + LLM call
# ---------------------------------------------------------------------------

class TestConsolidationPrompt:
    @pytest.mark.asyncio
    async def test_llm_call_returns_valid_semantic(self):
        client = _make_mock_client(can_consolidate=True, confidence=0.92)
        consolidation = DreamConsolidation(anthropic_client=client)
        cluster = [
            {"id": "1", "user_message": "Python tips?", "assistant_response": "Use comprehensions."},
            {"id": "2", "user_message": "Python best practices?", "assistant_response": "Avoid globals."},
        ]
        result = await consolidation._consolidate_cluster(cluster, client)
        assert result is not None
        assert "semantic_memory" in result
        assert result["confidence"] == pytest.approx(0.92, abs=0.01)

    @pytest.mark.asyncio
    async def test_non_consolidatable_cluster_returns_none(self):
        client = _make_mock_client(can_consolidate=False, confidence=0.5)
        consolidation = DreamConsolidation(anthropic_client=client)
        cluster = [
            {"id": "1", "user_message": "q1", "assistant_response": "a1"},
        ]
        result = await consolidation._consolidate_cluster(cluster, client)
        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_json_returns_none(self):
        client = MagicMock()
        response_obj = MagicMock()
        response_obj.content = [MagicMock()]
        response_obj.content[0].text = "not valid json at all!!!"
        client.messages = MagicMock()
        client.messages.create = AsyncMock(return_value=response_obj)

        consolidation = DreamConsolidation(anthropic_client=client)
        cluster = [{"id": "1", "user_message": "q", "assistant_response": "a"}]
        result = await consolidation._consolidate_cluster(cluster, client)
        assert result is None


# ---------------------------------------------------------------------------
# Provisional safeguard
# ---------------------------------------------------------------------------

class TestProvisionalSafeguard:
    @pytest.mark.asyncio
    async def test_low_confidence_skips_storage(self, tmp_path):
        db_url = f"sqlite+aiosqlite:///{tmp_path}/safeguard.db"
        log = ExperienceLogger(db_url=db_url)
        await log.init()
        store = SemanticStore(db_url=db_url)

        client = _make_mock_client(can_consolidate=True, confidence=0.50)  # below 0.80 threshold
        consolidation = DreamConsolidation(
            experience_logger=log,
            semantic_store=store,
            anthropic_client=client,
        )
        cluster = [{"id": "1", "user_message": "q", "assistant_response": "a"}]
        result = await consolidation._consolidate_cluster(cluster, client)
        # Should return skipped_low_confidence flag
        assert result is not None
        assert result.get("skipped_low_confidence") is True

        # Nothing stored
        confirmed = await store.get_confirmed_semantics()
        provisional = await store.get_provisional_semantics()
        assert confirmed == []
        assert provisional == []


# ---------------------------------------------------------------------------
# Validation and promotion
# ---------------------------------------------------------------------------

class TestValidationPromotion:
    @pytest.mark.asyncio
    async def test_validation_count_increments(self, tmp_path):
        db_url = f"sqlite+aiosqlite:///{tmp_path}/validation.db"
        log = ExperienceLogger(db_url=db_url)
        await log.init()
        store = SemanticStore(db_url=db_url)

        sem_id = await store.create_semantic_memory(
            content="User prefers Python",
            source_episode_ids=["ep1"],
            key_pattern="Python preference",
            confidence=0.9,
        )

        count = await store.increment_validation(sem_id)
        assert count == 1
        count = await store.increment_validation(sem_id)
        assert count == 2

    @pytest.mark.asyncio
    async def test_promotion_at_threshold(self, tmp_path):
        db_url = f"sqlite+aiosqlite:///{tmp_path}/promotion.db"
        log = ExperienceLogger(db_url=db_url)
        await log.init()
        store = SemanticStore(db_url=db_url)

        sem_id = await store.create_semantic_memory(
            content="User asks about Python a lot",
            source_episode_ids=["ep1"],
            key_pattern="Python pattern",
            confidence=0.9,
        )
        # Manually hit threshold
        for _ in range(3):
            await store.increment_validation(sem_id)
        await store.promote_to_confirmed(sem_id)

        confirmed = await store.get_confirmed_semantics()
        assert len(confirmed) == 1
        assert confirmed[0]["id"] == sem_id
        assert confirmed[0]["is_provisional"] is False
        assert confirmed[0]["fitness_score"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Pruning old unvalidated semantics
# ---------------------------------------------------------------------------

class TestOldUnvalidatedPruned:
    @pytest.mark.asyncio
    async def test_old_zero_validated_semantic_pruned(self, tmp_path):
        db_url = f"sqlite+aiosqlite:///{tmp_path}/prune.db"
        log = ExperienceLogger(db_url=db_url)
        await log.init()
        store = SemanticStore(db_url=db_url)

        sem_id = await store.create_semantic_memory(
            content="Old unvalidated memory",
            source_episode_ids=["ep1"],
            key_pattern="pattern",
            confidence=0.9,
        )

        # Manually backdate creation
        from sqlalchemy import update
        from data.experience_log import SemanticMemoryRow
        from datetime import timedelta
        async with store._session_factory() as session:
            await session.execute(
                update(SemanticMemoryRow)
                .where(SemanticMemoryRow.id == sem_id)
                .values(created_at=datetime.utcnow() - timedelta(days=20))
            )
            await session.commit()

        consolidation = DreamConsolidation(
            experience_logger=log,
            semantic_store=store,
            anthropic_client=_make_mock_client(),
        )
        promoted, pruned = await consolidation._validate_provisional_semantics(None)
        assert pruned == 1

        remaining = await store.get_provisional_semantics()
        assert remaining == []
