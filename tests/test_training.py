"""
Tests for DatasetBuilder and training data pipeline.

These tests use a real (temp) SQLite database but mock the
SFTTrainer to avoid needing a GPU.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from data.experience_log import ExperienceLogger
from training.data_builder import DatasetBuilder, _format_example


# ---------------------------------------------------------------------------
# _format_example
# ---------------------------------------------------------------------------

class TestFormatExample:
    def test_contains_chatml_markers(self):
        text = _format_example("Hello", "World")
        assert "<|im_start|>system" in text
        assert "<|im_start|>user" in text
        assert "<|im_start|>assistant" in text
        assert "<|im_end|>" in text
        assert "Hello" in text
        assert "World" in text


# ---------------------------------------------------------------------------
# DatasetBuilder
# ---------------------------------------------------------------------------

@pytest.fixture
async def full_logger(tmp_path):
    """Logger with enough episodes to pass TRAINING_MIN_EPISODES check."""
    db_url = f"sqlite+aiosqlite:///{tmp_path}/train_test.db"
    log = ExperienceLogger(db_url=db_url)
    await log.init()
    # Insert 60 high-fitness episodes (default min is 50)
    for i in range(60):
        eid = await log.log_episode(
            user_message=f"question {i}",
            assistant_response=f"answer {i}",
            full_prompt=f"full {i}",
            session_id="train_sess",
        )
        await log.update_fitness(eid, score=0.85, signal_type="explicit")
    return log


@pytest.fixture
async def sparse_logger(tmp_path):
    """Logger with fewer episodes than TRAINING_MIN_EPISODES."""
    db_url = f"sqlite+aiosqlite:///{tmp_path}/sparse_test.db"
    log = ExperienceLogger(db_url=db_url)
    await log.init()
    for i in range(5):
        eid = await log.log_episode(f"q{i}", f"a{i}", f"p{i}", "s")
        await log.update_fitness(eid, score=0.9, signal_type="explicit")
    return log


class TestDatasetBuilder:
    @pytest.mark.asyncio
    async def test_returns_none_when_not_enough_data(self, sparse_logger):
        builder = DatasetBuilder(experience_logger=sparse_logger, min_episodes=50)
        result = await builder.build()
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_dataset_when_enough_data(self, full_logger):
        builder = DatasetBuilder(experience_logger=full_logger, min_episodes=50)
        result = await builder.build()
        assert result is not None
        assert "train" in result
        assert "eval" in result

    @pytest.mark.asyncio
    async def test_train_eval_split(self, full_logger):
        builder = DatasetBuilder(experience_logger=full_logger, min_episodes=10)
        result = await builder.build()
        total = len(result["train"]) + len(result["eval"])
        assert total == 60
        # Eval should be ~10%
        assert len(result["eval"]) >= 4

    @pytest.mark.asyncio
    async def test_text_field_present(self, full_logger):
        builder = DatasetBuilder(experience_logger=full_logger, min_episodes=10)
        result = await builder.build()
        assert "text" in result["train"].column_names

    @pytest.mark.asyncio
    async def test_get_episode_ids_for_training(self, full_logger):
        builder = DatasetBuilder(experience_logger=full_logger, min_episodes=10)
        ids = await builder.get_episode_ids_for_training()
        assert len(ids) == 60
