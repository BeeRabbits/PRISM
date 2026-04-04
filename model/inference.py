"""
PrismInferenceEngine: runs generation with optional Titans memory adapter.

The engine:
  1. Retrieves the per-session Titans memory state.
  2. Registers a forward hook on the embedding layer that runs the adapter.
  3. Runs model.generate() with the hooked embeddings.
  4. Removes the hook and updates the session memory state.
  5. Returns the decoded response string.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

import config
from memory_adapter.memory_state import MemoryStateManager

logger = logging.getLogger(__name__)

# System prompt baked into every conversation
_DEFAULT_SYSTEM = (
    "You are PRISM, a persistent intelligent assistant. "
    "You learn from every conversation and grow over time."
)


class PrismInferenceEngine:
    """
    Wraps the loaded model + tokenizer and provides a single async
    generate() entry point.

    Args:
        model:           The (possibly PEFT-wrapped) causal LM.
        tokenizer:       The associated tokenizer.
        memory_manager:  MemoryStateManager for per-session Titans state.
    """

    def __init__(
        self,
        model,
        tokenizer,
        memory_manager: MemoryStateManager,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.memory_manager = memory_manager

    async def generate(
        self,
        session_id: str,
        message: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[dict]] = None,
    ) -> tuple[str, str]:
        """
        Run one inference turn.

        Args:
            session_id:           Session identifier for memory lookup.
            message:              The current user message.
            system_prompt:        Optional system prompt override.
            conversation_history: Prior turns [[role, content], ...].

        Returns:
            (response_text, full_prompt_text)
        """
        system = system_prompt or _DEFAULT_SYSTEM
        history = conversation_history or []

        # Inject validated semantic memories into system prompt
        if config.MEMORY_INJECTION_ENABLED:
            try:
                from data.semantic_store import get_semantic_store
                from data.memory_validator import get_memory_validator

                store = get_semantic_store()
                validator = get_memory_validator()

                confirmed = await store.get_confirmed_semantics(limit=10)
                if confirmed:
                    validated = await validator.validate_batch(confirmed)
                    if validated:
                        memory_lines = [f"- {m['content']}" for m in validated[:5]]
                        memory_block = "\n".join(memory_lines)
                        system = system + "\n\nConfirmed knowledge about this user:\n" + memory_block
            except Exception as e:
                logger.debug("Memory injection skipped: %s", e)

        # Build messages list in Qwen chat format
        messages: List[dict] = [{"role": "system", "content": system}]
        messages.extend(history)
        messages.append({"role": "user", "content": message})

        # Apply Qwen chat template
        device = next(self.model.parameters()).device
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        # apply_chat_template returns BatchEncoding in newer transformers versions
        if hasattr(tokenized, "input_ids"):
            input_ids = tokenized.input_ids.to(device)
        else:
            input_ids = tokenized.to(device)

        full_prompt = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)

        # Retrieve Titans session memory
        session_memory = await self.memory_manager.get_or_init(session_id)
        model_dtype = next(self.model.parameters()).dtype
        session_memory = session_memory.to(device=device, dtype=model_dtype)

        # Install Titans hook if adapter is available
        hook_handle = None
        updated_memory_container: list[Optional[torch.Tensor]] = [None]

        titans_adapter = getattr(self.model, "titans_adapter", None)
        if titans_adapter is not None:
            hook_handle, updated_memory_container = self._install_titans_hook(
                titans_adapter, session_memory
            )

        # Generate
        try:
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=1024,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        # Decode only the newly generated tokens
        new_tokens = output_ids[0][input_ids.shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Fix tokenizer artifacts: underscores replacing apostrophes in contractions
        # Handle both plain (_) and markdown-escaped (\_) underscores
        import re
        response = re.sub(
            r"(\w)\\?_(t|s|d|m|ll|re|ve)\b",
            r"\1'\2",
            response,
        )

        # Update session memory if adapter ran
        if updated_memory_container[0] is not None:
            await self.memory_manager.update(session_id, updated_memory_container[0])

        return response, full_prompt

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _install_titans_hook(
        self,
        titans_adapter,
        session_memory: torch.Tensor,
    ):
        """
        Register a forward hook on the embedding layer that passes
        hidden states through the Titans adapter.

        Returns (hook_handle, updated_memory_container).
        The hook writes the updated memory into updated_memory_container[0].
        """
        updated_memory_container: list[Optional[torch.Tensor]] = [None]

        def _hook(module, args, output):
            # output is (hidden_states,) or just hidden_states depending on model
            hidden = output[0] if isinstance(output, tuple) else output
            enhanced, new_mem = titans_adapter(hidden, session_memory=session_memory)
            updated_memory_container[0] = new_mem
            if isinstance(output, tuple):
                return (enhanced,) + output[1:]
            return enhanced

        # Find the embedding layer (model may be wrapped in PeftModel or bare)
        base = self.model
        # Unwrap PeftModel → LoraModel → original model
        if hasattr(base, "base_model"):
            base = base.base_model
        if hasattr(base, "model") and hasattr(base.model, "model"):
            base = base.model
        # Now base should be the CausalLM; get embed_tokens from its inner model
        if hasattr(base, "model") and hasattr(base.model, "embed_tokens"):
            embed_layer = base.model.embed_tokens
        elif hasattr(base, "embed_tokens"):
            embed_layer = base.embed_tokens
        else:
            raise AttributeError(f"Cannot find embed_tokens on model: {type(base)}")

        handle = embed_layer.register_forward_hook(_hook)
        return handle, updated_memory_container
