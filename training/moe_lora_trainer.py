"""
MoE-LoRA Trainer: trains individual expert LoRA adapters.

Unlike the original LoRAContinualTrainer which trained one adapter on
all personal data (causing catastrophic forgetting), MoE-LoRA trains
separate, focused experts on carefully curated data.

Current experts:
  - "style": Trained on episodes where the oracle scored 4+ and delta < 1.0.
             These represent moments when PRISM performed well — the model
             learns HOW to respond, not WHAT facts to remember.

Training pipeline:
  1. EvalRunner captures baseline (pre-training model state)
  2. Expert data is curated from high-quality episodes
  3. LoRA trained with conservative hyperparameters
  4. EvalRunner runs post-training evaluation
  5. If general capability regresses → adapter rolled back
  6. If passes → adapter saved, hot-swapped with low weight

Neuroscience parallel:
  This is the "slow neocortical update" from CLS theory. Unlike the
  hippocampus (knowledge graph) which learns fast, the neocortex (weights)
  updates slowly, only from extensively validated patterns, and only in
  specific regions (experts) — not globally.
"""

from __future__ import annotations

import gc
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset, DatasetDict
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer, SFTConfig

import config
from data.experience_log import get_experience_logger
from evaluation.eval_runner import EvalRunner
from model.expert_router import EXPERTS

logger = logging.getLogger(__name__)

# Style expert: very conservative hyperparameters
_STYLE_LR = 2e-5  # Very low — preserving base capability is priority
_STYLE_RANK = 8  # Low rank — style is a small behavioral shift
_STYLE_ALPHA = 16
_STYLE_MODULES = ["q_proj", "v_proj"]  # Minimal footprint
_STYLE_MIN_EPISODES = 100  # Need enough high-quality examples
_STYLE_MAX_EPISODES = 500  # Cap to prevent overfitting


class MoELoRATrainer:
    """
    Trains individual expert LoRA adapters with evaluation gates.

    Args:
        model:     The base model (must NOT be PEFT-wrapped).
        tokenizer: The associated tokenizer.
    """

    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self._log = get_experience_logger()

    async def train_style_expert(self) -> dict:
        """
        Train the style expert adapter.

        Style data: episodes where oracle_score >= 4 and mirror_delta <= 1.0.
        These are moments when PRISM's response was high quality and the
        model's self-assessment was close to the oracle's judgment.

        Returns:
            dict with status, metrics, eval results.
        """
        logger.info("=== MoE-LoRA: training style expert ===")

        # Step 1: Curate style training data
        dataset = await self._build_style_dataset()
        if dataset is None:
            return {"status": "skipped", "reason": "not_enough_style_data"}

        # Step 2: Capture baseline evaluation
        eval_runner = EvalRunner(self.model, self.tokenizer)
        baseline = await eval_runner.run_evaluation(label="pre_style_training")

        # Step 3: Train
        expert_config = EXPERTS["style"]
        adapter_path = Path(expert_config.adapter_path)
        adapter_path.mkdir(parents=True, exist_ok=True)

        try:
            metrics = self._train_expert(
                dataset=dataset,
                output_path=adapter_path,
                learning_rate=_STYLE_LR,
                rank=_STYLE_RANK,
                alpha=_STYLE_ALPHA,
                target_modules=_STYLE_MODULES,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("CUDA OOM during style expert training.")
                return {"status": "oom", "error": str(e)}
            raise

        # Step 4: Load the new adapter and evaluate
        try:
            test_model = PeftModel.from_pretrained(
                self.model,
                str(adapter_path),
                is_trainable=False,
                torch_dtype=torch.bfloat16,
            )
            eval_runner_post = EvalRunner(test_model, self.tokenizer)
            post_eval = await eval_runner_post.run_evaluation(label="post_style_training")

            # Unload test model
            test_model = test_model.unload()
            if hasattr(test_model, "peft_config"):
                delattr(test_model, "peft_config")
            del test_model
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error("Post-training evaluation failed: %s", e)
            return {"status": "eval_failed", "error": str(e), "metrics": metrics}

        # Step 5: Compare and decide
        comparison = eval_runner.compare_to_baseline(post_eval)

        if comparison.get("regression_detected", False):
            # ROLLBACK: delete the adapter
            logger.warning("Style expert FAILED eval gate. Rolling back adapter.")
            import shutil
            shutil.rmtree(adapter_path, ignore_errors=True)
            return {
                "status": "rolled_back",
                "reason": "general_capability_regression",
                "metrics": metrics,
                "eval_baseline": baseline,
                "eval_post": post_eval,
                "comparison": comparison,
            }

        logger.info("Style expert PASSED eval gate. Adapter saved to %s", adapter_path)
        return {
            "status": "completed",
            "metrics": metrics,
            "eval_baseline": baseline,
            "eval_post": post_eval,
            "comparison": comparison,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _build_style_dataset(self) -> Optional[DatasetDict]:
        """Build training data for the style expert from high-quality episodes."""
        episodes = await self._log.get_high_oracle_episodes(
            min_oracle_score=4.0,
            max_delta=1.0,
            limit=_STYLE_MAX_EPISODES,
        )

        if len(episodes) < _STYLE_MIN_EPISODES:
            logger.info(
                "Style expert: not enough high-quality episodes (%d/%d).",
                len(episodes), _STYLE_MIN_EPISODES,
            )
            return None

        logger.info("Style expert: %d high-quality episodes selected.", len(episodes))

        system_prompt = (
            "You are PRISM, a persistent intelligent assistant. "
            "You learn from every conversation and grow over time."
        )

        texts = []
        for ep in episodes:
            text = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{ep['user_message']}<|im_end|>\n"
                f"<|im_start|>assistant\n{ep['assistant_response']}<|im_end|>"
            )
            texts.append(text)

        raw = Dataset.from_dict({"text": texts})
        split = raw.train_test_split(test_size=0.1, seed=42)
        return DatasetDict({"train": split["train"], "eval": split["test"]})

    def _train_expert(
        self,
        dataset: DatasetDict,
        output_path: Path,
        learning_rate: float,
        rank: int,
        alpha: int,
        target_modules: list,
    ) -> dict:
        """Run SFTTrainer for an expert adapter."""
        from peft import PeftModel as PM

        # Ensure clean base model
        base_model = self.model
        if isinstance(base_model, PM):
            base_model = base_model.unload()
            self.model = base_model
        if hasattr(base_model, "peft_config"):
            delattr(base_model, "peft_config")

        gc.collect()
        torch.cuda.empty_cache()

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        peft_model = get_peft_model(base_model, lora_config)
        peft_model.print_trainable_parameters()

        sft_config = SFTConfig(
            per_device_train_batch_size=config.BATCH_SIZE,
            gradient_accumulation_steps=config.GRAD_ACCUM_STEPS,
            warmup_ratio=0.05,
            num_train_epochs=1,
            learning_rate=learning_rate,
            fp16=False,
            bf16=True,
            logging_steps=10,
            output_dir=str(output_path),
            optim="adamw_8bit",
            save_strategy="no",
            eval_strategy="epoch",
            load_best_model_at_end=False,
            report_to="none",
            max_seq_length=config.MAX_SEQ_LENGTH,
            dataset_text_field="text",
            packing=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

        trainer = SFTTrainer(
            model=peft_model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["eval"],
            args=sft_config,
        )

        try:
            train_result = trainer.train()
            eval_result = trainer.evaluate()
        except Exception:
            logger.warning("Expert training failed. Cleaning up...")
            try:
                clean = peft_model.unload()
                self.model = clean
            except Exception:
                pass
            del trainer
            gc.collect()
            torch.cuda.empty_cache()
            raise

        # Save the expert adapter
        trainer.save_model(str(output_path))
        logger.info("Expert adapter saved to %s", output_path)

        # Clean up
        del trainer
        clean_base = peft_model.unload()
        if hasattr(clean_base, "peft_config"):
            delattr(clean_base, "peft_config")
        self.model = clean_base
        gc.collect()
        torch.cuda.empty_cache()

        return {
            "train_loss": train_result.training_loss,
            "eval_loss": eval_result.get("eval_loss"),
        }
