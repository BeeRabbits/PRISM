"""
Grafted Titans Memory Adapter: TitansMemoryAdapter(nn.Module)

Architecture:
  1. A persistent memory bank: nn.Parameter [memory_size, memory_dim]
  2. Cross-attention from current hidden states → memory bank
  3. MaG gating to blend memory context into hidden states
  4. Online memory update rule (Titans-inspired, momentum=0.9)

The adapter is injected at Layer 0 of the transformer as a
forward hook on the embedding layer. The base model weights
are NEVER modified. Only the adapter parameters train.

Forward pass:
    enhanced_hidden_states, updated_memory = adapter(hidden_states)

Memory update (Titans online rule):
    new_memory = momentum * old_memory
                 + (1 - momentum) * outer_product_update(hidden_states)

The "outer product update" here is a simplified Hebbian-style
association between query projections and the mean hidden state,
which is a tractable approximation of the full Titans gradient update.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from memory_adapter.gating import MemoryAsGate
import config


class TitansMemoryAdapter(nn.Module):
    """
    Titans-inspired memory adapter grafted onto a transformer's embedding output.

    Args:
        model_hidden_size: Hidden dimension of the base transformer (e.g. 3584 for Qwen2.5-7B).
        memory_size:       Number of memory slots (rows in the bank).
        memory_dim:        Dimension of each memory slot.
        momentum:          EMA coefficient for online memory update.
        num_heads:         Number of attention heads in cross-attention.
    """

    def __init__(
        self,
        model_hidden_size: int,
        memory_size: int = 512,
        memory_dim: int = 512,
        momentum: float = 0.9,
        num_heads: int = 8,
    ) -> None:
        super().__init__()

        self.model_hidden_size = model_hidden_size
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.momentum = momentum
        self.num_heads = num_heads
        self.head_dim = memory_dim // num_heads

        assert memory_dim % num_heads == 0, (
            f"memory_dim ({memory_dim}) must be divisible by num_heads ({num_heads})"
        )

        # ── Persistent memory bank ──────────────────────────────────────────
        # This is the "long-term memory" that persists across sessions
        # when saved to disk. Initialised with small values.
        self.memory_bank = nn.Parameter(
            torch.randn(memory_size, memory_dim) * 0.01
        )

        # ── Projection layers ────────────────────────────────────────────────
        # Project model hidden states down to memory_dim for cross-attention
        self.q_proj = nn.Linear(model_hidden_size, memory_dim, bias=False)
        self.k_proj = nn.Linear(memory_dim, memory_dim, bias=False)
        self.v_proj = nn.Linear(memory_dim, memory_dim, bias=False)
        self.out_proj = nn.Linear(memory_dim, model_hidden_size, bias=False)

        # ── Memory update projection ─────────────────────────────────────────
        # Maps hidden states to memory_dim for the update rule
        self.update_proj = nn.Linear(model_hidden_size, memory_dim, bias=False)

        # ── MaG gating ───────────────────────────────────────────────────────
        self.gate = MemoryAsGate(
            hidden_size=model_hidden_size,
            memory_dim=model_hidden_size,  # out_proj brings context back to hidden_size
        )

        # ── Layer norm for stability ─────────────────────────────────────────
        self.layer_norm = nn.LayerNorm(model_hidden_size)

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """
        Safe initialisation:
        - Projection layers get Xavier init (for when training starts).
        - out_proj starts at zero so the adapter is a complete no-op on
          a fresh bootstrap. Combined with the gate bias=-5 in MaG, a
          new adapter produces: enhanced ≈ layernorm(0 + hidden_states)
          = hidden_states (since layernorm is identity for single-scale).
          Training gradually opens both the gate and the projections.
        """
        for module in [self.q_proj, self.k_proj, self.v_proj, self.update_proj]:
            nn.init.xavier_uniform_(module.weight)
        # out_proj starts at zero → memory_context = 0 → gate output = 0*0 + (1-0)*h = h
        nn.init.zeros_(self.out_proj.weight)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        session_memory: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the Titans memory adapter forward pass.

        Args:
            hidden_states:  [batch, seq_len, model_hidden_size]
                            The embedding layer output.
            session_memory: Optional [memory_size, memory_dim] per-session
                            memory tensor. If None, uses the persistent
                            memory_bank parameter.
            attention_mask: [batch, seq_len], unused for memory attention
                            but kept for API compatibility.

        Returns:
            enhanced_hidden_states: [batch, seq_len, model_hidden_size]
            updated_memory:         [memory_size, memory_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Use session-specific memory if provided, else the persistent bank
        active_memory = session_memory if session_memory is not None else self.memory_bank
        # Expand for batch: [1, memory_size, memory_dim] → [B, memory_size, memory_dim]
        active_memory_expanded = active_memory.unsqueeze(0).expand(batch_size, -1, -1)

        # ── Cross-attention: hidden states query the memory bank ──────────
        memory_context = self._cross_attend(hidden_states, active_memory_expanded)
        # memory_context: [B, seq_len, model_hidden_size]

        # ── MaG gating ───────────────────────────────────────────────────
        # Gate output: gate * memory_context + (1-gate) * hidden_states
        # With fresh init: gate≈0, out_proj=0 → output ≈ hidden_states
        enhanced = self.gate(hidden_states, memory_context)

        # ── Online memory update ─────────────────────────────────────────
        updated_memory = self._update_memory(hidden_states, active_memory)

        return enhanced, updated_memory

    # ------------------------------------------------------------------
    # Cross-attention
    # ------------------------------------------------------------------

    def _cross_attend(
        self,
        hidden_states: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Multi-head cross-attention from hidden states to memory bank.

        Q: projected hidden states  [B, S, memory_dim]
        K: projected memory         [B, M, memory_dim]
        V: projected memory         [B, M, memory_dim]

        Returns context projected back to [B, S, model_hidden_size].
        """
        B, S, _ = hidden_states.shape
        M = memory.shape[1]
        H = self.num_heads
        D = self.head_dim

        Q = self.q_proj(hidden_states)   # [B, S, memory_dim]
        K = self.k_proj(memory)          # [B, M, memory_dim]
        V = self.v_proj(memory)          # [B, M, memory_dim]

        # Reshape for multi-head attention
        Q = Q.view(B, S, H, D).transpose(1, 2)   # [B, H, S, D]
        K = K.view(B, M, H, D).transpose(1, 2)   # [B, H, M, D]
        V = V.view(B, M, H, D).transpose(1, 2)   # [B, H, M, D]

        # Scaled dot-product attention
        scale = math.sqrt(D)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # [B, H, S, M]
        weights = F.softmax(scores, dim=-1)

        context = torch.matmul(weights, V)          # [B, H, S, D]
        context = context.transpose(1, 2).contiguous().view(B, S, -1)  # [B, S, memory_dim]

        # Project back to model hidden size
        return self.out_proj(context)  # [B, S, model_hidden_size]

    # ------------------------------------------------------------------
    # Online memory update (Titans-inspired)
    # ------------------------------------------------------------------

    def _update_memory(
        self,
        hidden_states: torch.Tensor,
        current_memory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Titans-inspired online memory update:

            new_memory = momentum * old_memory
                       + (1 - momentum) * outer_product_update(hidden_states)

        The outer product update is computed as:
            projected_h = mean(update_proj(hidden_states), dim=1)  # [B, memory_dim]
            delta = mean over batch of (projected_h^T @ projected_h normalised)

        This is a tractable Hebbian-style associative update that pushes
        the memory bank towards the principal directions of the current input.

        Args:
            hidden_states:  [B, seq_len, model_hidden_size]
            current_memory: [memory_size, memory_dim]  (no batch dim)

        Returns:
            updated_memory: [memory_size, memory_dim]
        """
        # Project hidden states to memory_dim
        projected = self.update_proj(hidden_states)   # [B, S, memory_dim]
        mean_repr = projected.mean(dim=1)             # [B, memory_dim]

        # Average over batch
        mean_repr = mean_repr.mean(dim=0)             # [memory_dim]

        # Outer product: [memory_size, memory_dim] update signal
        # We compute a rank-1 update: mean_repr.unsqueeze(0) is broadcast
        # over the memory_size dimension.
        delta = mean_repr.unsqueeze(0).expand_as(current_memory)  # [memory_size, memory_dim]
        # L2-normalise the delta for stability
        delta = F.normalize(delta, dim=-1)

        updated = self.momentum * current_memory + (1.0 - self.momentum) * delta
        return updated.detach()  # stop gradient through memory update
