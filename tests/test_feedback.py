"""
Tests for the /feedback endpoint and ExperienceLogger fitness updates.
"""

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from data.experience_log import ExperienceLogger


@pytest.fixture
async def logger(tmp_path):
    """Create a fresh ExperienceLogger backed by a temp SQLite database."""
    db_url = f"sqlite+aiosqlite:///{tmp_path}/test.db"
    log = ExperienceLogger(db_url=db_url)
    await log.init()
    return log


@pytest.fixture
async def seeded_episode(logger):
    """Insert one episode and return its ID."""
    episode_id = await logger.log_episode(
        user_message="hello",
        assistant_response="hi there",
        full_prompt="<full prompt>",
        session_id="test_session",
    )
    return episode_id


class TestExperienceLogger:
    @pytest.mark.asyncio
    async def test_log_episode_returns_uuid(self, logger):
        eid = await logger.log_episode(
            user_message="test",
            assistant_response="response",
            full_prompt="prompt",
            session_id="s1",
        )
        assert isinstance(eid, str)
        uuid.UUID(eid)  # should not raise

    @pytest.mark.asyncio
    async def test_update_fitness_first_score_direct(self, logger, seeded_episode):
        new_fitness = await logger.update_fitness(
            episode_id=seeded_episode,
            score=0.8,
            signal_type="explicit",
        )
        assert abs(new_fitness - 0.8) < 1e-6

    @pytest.mark.asyncio
    async def test_update_fitness_ema(self, logger, seeded_episode):
        # First update sets score directly
        await logger.update_fitness(seeded_episode, score=0.8, signal_type="explicit")
        # Second update applies EMA: 0.7 * 0.8 + 0.3 * 0.4 = 0.68
        new_fitness = await logger.update_fitness(seeded_episode, score=0.4, signal_type="explicit")
        assert abs(new_fitness - (0.7 * 0.8 + 0.3 * 0.4)) < 1e-5

    @pytest.mark.asyncio
    async def test_unknown_episode_raises(self, logger):
        with pytest.raises(ValueError, match="Episode not found"):
            await logger.update_fitness(
                episode_id=str(uuid.uuid4()),
                score=0.5,
                signal_type="explicit",
            )

    @pytest.mark.asyncio
    async def test_get_high_fitness_episodes(self, logger):
        # Log 3 episodes with different fitness scores
        e1 = await logger.log_episode("q1", "a1", "p1", "s1")
        e2 = await logger.log_episode("q2", "a2", "p2", "s1")
        e3 = await logger.log_episode("q3", "a3", "p3", "s1")

        await logger.update_fitness(e1, score=0.9, signal_type="explicit")
        await logger.update_fitness(e2, score=0.3, signal_type="explicit")
        # e3 stays at default fitness 1.0

        episodes = await logger.get_high_fitness_episodes(min_score=0.6)
        ids = {ep["id"] for ep in episodes}
        assert e1 in ids
        assert e3 in ids
        assert e2 not in ids

    @pytest.mark.asyncio
    async def test_mark_training_used(self, logger, seeded_episode):
        await logger.mark_training_used([seeded_episode])
        # After marking, should not appear in untrained episodes
        untrained = await logger.get_untrained_episodes(min_score=0.0)
        ids = {ep["id"] for ep in untrained}
        assert seeded_episode not in ids

    @pytest.mark.asyncio
    async def test_count_total(self, logger):
        assert await logger.count_total() == 0
        await logger.log_episode("a", "b", "c", "s")
        await logger.log_episode("d", "e", "f", "s")
        assert await logger.count_total() == 2
