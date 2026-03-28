"""
TitansAdapterTrainer: trains only the Titans memory adapter parameters.

The base model and LoRA adapters are kept frozen.
The Titans adapter is trained on the same high-fitness episode data
used for LoRA fine-tuning, using cross-entropy loss on next-token
prediction with the adapter enabled.

The trained adapter is saved to TITANS_ADAPTER_PATH/titans_adapter.pt.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

import config
from data.experience_log import ExperienceLogger, get_experience_logger
from training.data_builder import DatasetBuilder, _format_example

logger = logging.getLogger(__name__)


class TitansAdapterTrainer:
    """
    Trains only the TitansMemoryAdapter parameters.

    Args:
        model:             The loaded causal LM (with adapter attached as .titans_adapter).
        tokenizer:         The associated tokenizer.
        experience_logger: Source of training episodes.
    """

    def __init__(
        self,
        model,
        tokenizer,
        experience_logger: Optional[ExperienceLogger] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self._log = experience_logger or get_experience_logger()
        self._builder = DatasetBuilder(experience_logger=self._log)

    def train(self) -> dict:
        """
        Run one training pass on the Titans adapter.

        Returns:
            dict with status, loss, and checkpoint path.
        """
        titans = getattr(self.model, "titans_adapter", None)
        if titans is None:
            logger.warning("No titans_adapter found on model. Skipping Titans training.")
            return {"status": "skipped", "reason": "no adapter"}

        started_at = datetime.utcnow()
        import asyncio
        loop = asyncio.new_event_loop()
        episodes = loop.run_until_complete(
            self._log.get_all_high_fitness_episodes(
                min_score=config.TRAINING_MIN_FITNESS,
                limit=500,
            )
        )
        loop.close()

        if len(episodes) < config.TRAINING_MIN_EPISODES:
            logger.warning(
                "Not enough episodes for Titans training: %d / %d",
                len(episodes),
                config.TRAINING_MIN_EPISODES,
            )
            return {"status": "skipped", "reason": "not_enough_data"}

        logger.info("Training Titans adapter on %d episodes.", len(episodes))

        # Freeze everything except the Titans adapter
        for param in self.model.parameters():
            param.requires_grad = False
        for param in titans.parameters():
            param.requires_grad = True

        optimizer = AdamW(titans.parameters(), lr=1e-4)
        total_loss = 0.0
        steps = 0

        titans.train()

        for ep in episodes:
            text = _format_example(ep["user_message"], ep["assistant_response"])
            try:
                loss = self._train_step(titans, optimizer, text)
                total_loss += loss
                steps += 1
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("OOM on episode %s. Skipping.", ep["id"])
                    torch.cuda.empty_cache()
                    continue
                raise

        titans.eval()

        avg_loss = total_loss / max(steps, 1)
        checkpoint = self._save(titans)
        logger.info(
            "Titans adapter training complete. steps=%d avg_loss=%.4f saved=%s",
            steps, avg_loss, checkpoint,
        )
        return {
            "status": "completed",
            "steps": steps,
            "avg_loss": avg_loss,
            "checkpoint": str(checkpoint),
            "started_at": started_at.isoformat(),
            "completed_at": datetime.utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_step(
        self,
        titans: nn.Module,
        optimizer: AdamW,
        text: str,
    ) -> float:
        """
        One gradient step using next-token prediction loss.

        We:
          1. Tokenize the text.
          2. Run model forward with the Titans hook active.
          3. Compute cross-entropy loss on next tokens.
          4. Backprop only through the Titans adapter parameters.
        """
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,  # keep short for training efficiency
        )
        input_ids = enc["input_ids"].to(self.model.device)

        if input_ids.shape[1] < 2:
            return 0.0

        labels = input_ids.clone()

        # Install hook to route embeddings through Titans
        hook_handle = None

        def _hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # Init zeros memory for training (no session state)
            mem = torch.zeros(
                config.TITANS_MEMORY_SIZE, config.TITANS_MEMORY_DIM,
                device=hidden.device, dtype=hidden.dtype
            )
            enhanced, _ = titans(hidden, session_memory=mem)
            if isinstance(output, tuple):
                return (enhanced,) + output[1:]
            return enhanced

        # Find embed_tokens (handles both PeftModel-wrapped and bare models)
        base = self.model
        if hasattr(base, "base_model"):
            base = base.base_model
        if hasattr(base, "model") and hasattr(base.model, "model"):
            base = base.model
        if hasattr(base, "model") and hasattr(base.model, "embed_tokens"):
            embed_layer = base.model.embed_tokens
        elif hasattr(base, "embed_tokens"):
            embed_layer = base.embed_tokens
        else:
            raise AttributeError(f"Cannot find embed_tokens on model: {type(base)}")
        hook_handle = embed_layer.register_forward_hook(_hook)

        try:
            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(titans.parameters(), max_norm=1.0)
            optimizer.step()
        finally:
            hook_handle.remove()

        return loss.item()

    def _save(self, titans: nn.Module) -> Path:
        """Save the Titans adapter state dict to disk.

        When LoRA is active, PEFT wraps linear layers so keys become
        'q_proj.base_layer.weight' instead of 'q_proj.weight'.
        We strip the LoRA wrapping so the checkpoint can always be
        loaded into a fresh (unwrapped) TitansMemoryAdapter.
        """
        out_dir = Path(config.TITANS_ADAPTER_PATH)
        out_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = out_dir / "titans_adapter.pt"

        # Clean state dict: keep only base layer weights, strip LoRA keys
        raw_state = titans.state_dict()
        clean_state = {}
        for key, value in raw_state.items():
            if ".lora_A." in key or ".lora_B." in key:
                continue  # skip LoRA adapter weights
            # Strip '.base_layer' from PEFT-wrapped keys
            clean_key = key.replace(".base_layer", "")
            clean_state[clean_key] = value

        torch.save(clean_state, str(checkpoint))
        return checkpoint
