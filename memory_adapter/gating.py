"""
Memory-as-Gate (MaG) gating mechanism.

The gate controls how much memory context is blended into
the current hidden states:

    gate   = sigmoid(W_g @ hidden_states)
    output = gate * memory_context + (1 - gate) * hidden_states

This lets the model learn when to trust the memory bank and
when to rely on the raw token representations.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MemoryAsGate(nn.Module):
    """
    Learnable gate that blends memory context into hidden states.

    Args:
        hidden_size: Dimension of transformer hidden states.
        memory_dim:  Dimension of the memory bank vectors.
                     If different from hidden_size, a projection is applied
                     before gating so dimensions align.
    """

    def __init__(self, hidden_size: int, memory_dim: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_dim = memory_dim

        # Gate projection: maps hidden states → gate weights
        # Bias initialized to -5.0 so sigmoid(-5) ≈ 0.007, meaning the gate
        # starts nearly closed: output ≈ hidden_states (no memory corruption).
        # Training gradually opens the gate as Titans learns useful patterns.
        self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -5.0)

        # If memory_dim != hidden_size we need to project memory context
        if memory_dim != hidden_size:
            self.memory_proj: nn.Module = nn.Linear(memory_dim, hidden_size, bias=False)
        else:
            self.memory_proj = nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply MaG gating.

        Args:
            hidden_states:  [batch, seq_len, hidden_size], raw token embeddings.
            memory_context: [batch, seq_len, memory_dim], compressed memory signal.

        Returns:
            gated_output:   [batch, seq_len, hidden_size]
        """
        # Project memory context to hidden_size if needed
        projected_memory = self.memory_proj(memory_context)  # [B, S, H]

        # Compute gate from hidden states
        gate = torch.sigmoid(self.gate_proj(hidden_states))  # [B, S, H]

        # Blend
        output = gate * projected_memory + (1.0 - gate) * hidden_states
        return output
