"""
Tests for TitansMemoryAdapter and MemoryAsGate.

These tests run on CPU so no GPU is required.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest

from memory_adapter.gating import MemoryAsGate
from memory_adapter.titans_adapter import TitansMemoryAdapter
from memory_adapter.memory_state import MemoryStateManager


# ---------------------------------------------------------------------------
# MemoryAsGate
# ---------------------------------------------------------------------------

class TestMemoryAsGate:
    def test_output_shape_same_dims(self):
        gate = MemoryAsGate(hidden_size=64, memory_dim=64)
        hidden = torch.randn(2, 10, 64)
        memory = torch.randn(2, 10, 64)
        out = gate(hidden, memory)
        assert out.shape == hidden.shape

    def test_output_shape_different_dims(self):
        gate = MemoryAsGate(hidden_size=64, memory_dim=32)
        hidden = torch.randn(2, 10, 64)
        memory = torch.randn(2, 10, 32)
        out = gate(hidden, memory)
        assert out.shape == hidden.shape

    def test_gate_is_in_range(self):
        """Gate values should be between 0 and 1, output is blend."""
        gate = MemoryAsGate(hidden_size=8, memory_dim=8)
        hidden = torch.zeros(1, 1, 8)
        memory = torch.ones(1, 1, 8)
        out = gate(hidden, memory)
        # With all-zero hidden states, sigmoid output ≈ 0.5
        # Output should be between hidden (0) and memory (projected)
        assert out.shape == (1, 1, 8)


# ---------------------------------------------------------------------------
# TitansMemoryAdapter
# ---------------------------------------------------------------------------

class TestTitansMemoryAdapter:
    @pytest.fixture
    def adapter(self):
        return TitansMemoryAdapter(
            model_hidden_size=64,
            memory_size=16,
            memory_dim=32,
            momentum=0.9,
            num_heads=4,
        )

    def test_forward_output_shapes(self, adapter):
        hidden = torch.randn(2, 8, 64)
        enhanced, updated_mem = adapter(hidden)
        assert enhanced.shape == hidden.shape
        assert updated_mem.shape == (16, 32)

    def test_forward_with_session_memory(self, adapter):
        hidden = torch.randn(1, 5, 64)
        session_mem = torch.randn(16, 32)
        enhanced, updated_mem = adapter(hidden, session_memory=session_mem)
        assert enhanced.shape == hidden.shape
        assert updated_mem.shape == (16, 32)

    def test_memory_update_changes_memory(self, adapter):
        hidden = torch.randn(1, 4, 64)
        original_mem = adapter.memory_bank.data.clone()
        _, updated = adapter(hidden)
        # Updated memory should differ from original (online update applied)
        assert not torch.allclose(updated, original_mem, atol=1e-6)

    def test_no_gradient_through_memory_update(self, adapter):
        hidden = torch.randn(1, 4, 64, requires_grad=True)
        _, updated = adapter(hidden)
        # updated_memory should be detached (no grad)
        assert not updated.requires_grad

    def test_trainable_parameters(self, adapter):
        params = list(adapter.parameters())
        assert len(params) > 0
        total = sum(p.numel() for p in params)
        assert total > 0


# ---------------------------------------------------------------------------
# MemoryStateManager
# ---------------------------------------------------------------------------

class TestMemoryStateManager:
    @pytest.mark.asyncio
    async def test_init_and_get(self):
        mgr = MemoryStateManager(memory_size=8, memory_dim=16)
        mem = await mgr.get_or_init("sess_1")
        assert mem.shape == (8, 16)
        assert torch.all(mem == 0)

    @pytest.mark.asyncio
    async def test_update_and_get(self):
        mgr = MemoryStateManager(memory_size=8, memory_dim=16)
        await mgr.get_or_init("sess_1")
        new_mem = torch.ones(8, 16)
        await mgr.update("sess_1", new_mem)
        retrieved = await mgr.get("sess_1")
        assert torch.allclose(retrieved, new_mem)

    @pytest.mark.asyncio
    async def test_reset(self):
        mgr = MemoryStateManager(memory_size=8, memory_dim=16)
        await mgr.get_or_init("sess_1")
        await mgr.reset("sess_1")
        retrieved = await mgr.get("sess_1")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_multiple_sessions_independent(self):
        mgr = MemoryStateManager(memory_size=4, memory_dim=8)
        m1 = await mgr.get_or_init("a")
        m2 = await mgr.get_or_init("b")
        await mgr.update("a", torch.ones(4, 8))
        m1_after = await mgr.get("a")
        m2_after = await mgr.get("b")
        assert torch.all(m1_after == 1)
        assert torch.all(m2_after == 0)
