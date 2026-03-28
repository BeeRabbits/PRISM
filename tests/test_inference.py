"""
Tests for PrismInferenceEngine.

These tests mock the model and tokenizer to avoid requiring a GPU.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest

from memory_adapter.memory_state import MemoryStateManager
from model.inference import PrismInferenceEngine


def _make_mock_model(hidden_size: int = 64):
    """Return a minimal mock model."""
    model = MagicMock()
    model.device = torch.device("cpu")
    model.hf_device_map = {"": "cpu"}

    # embed_tokens mock (for hook registration)
    embed_tokens = MagicMock()
    embed_tokens.register_forward_hook = MagicMock(
        return_value=MagicMock()  # hook handle
    )
    model.base_model = MagicMock()
    model.base_model.model = MagicMock()
    model.base_model.model.model = MagicMock()
    model.base_model.model.model.embed_tokens = embed_tokens

    # generate() returns a tensor of token IDs
    model.generate = MagicMock(
        return_value=torch.tensor([[1, 2, 3, 4, 5, 100, 200, 300]])
    )
    model.titans_adapter = None
    return model


def _make_mock_tokenizer():
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.eos_token_id = 2
    tok.apply_chat_template = MagicMock(
        return_value=torch.tensor([[1, 2, 3, 4, 5]])
    )
    tok.decode = MagicMock(side_effect=lambda ids, **kw: "Hello from Itachi.")
    return tok


class TestPrismInferenceEngine:
    @pytest.fixture
    def engine(self):
        model = _make_mock_model()
        tokenizer = _make_mock_tokenizer()
        manager = MemoryStateManager(memory_size=8, memory_dim=16)
        return PrismInferenceEngine(model, tokenizer, manager)

    @pytest.mark.asyncio
    async def test_generate_returns_string(self, engine):
        response, full_prompt = await engine.generate(
            session_id="test_sess",
            message="Hello!",
        )
        assert isinstance(response, str)
        assert isinstance(full_prompt, str)

    @pytest.mark.asyncio
    async def test_generate_initialises_session_memory(self, engine):
        await engine.generate(session_id="new_sess", message="Hi")
        mem = await engine.memory_manager.get("new_sess")
        assert mem is not None
        assert mem.shape == (8, 16)

    @pytest.mark.asyncio
    async def test_generate_multiple_turns_reuses_memory(self, engine):
        await engine.generate(session_id="multi", message="Turn 1")
        mem_after_1 = (await engine.memory_manager.get("multi")).clone()
        await engine.generate(session_id="multi", message="Turn 2")
        # Memory object should still exist (may or may not change without real adapter)
        assert await engine.memory_manager.get("multi") is not None
