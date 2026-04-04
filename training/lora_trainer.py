"""
LoRAContinualTrainer. QLoRA fine-tuning pipeline.

Responsibilities:
  - Apply LoRA config to the base model (if not already applied).
  - Run SFTTrainer on high-fitness episode data.
  - Save the new adapter.
  - Mark trained episodes in ExperienceLogger.
  - Log training run metadata to data/training_runs.jsonl.
  - Handle CUDA OOM gracefully (halve batch size and retry once).
"""

from __future__ import annotations

import gc
import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from datasets import DatasetDict
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

import config
from data.experience_log import ExperienceLogger, get_experience_logger
from training.data_builder import DatasetBuilder

logger = logging.getLogger(__name__)

_RUNS_LOG = Path("data/training_runs.jsonl")

# Global lock so only one training run executes at a time
_training_lock = threading.Lock()


class LoRAContinualTrainer:
    """
    Continual fine-tuner that updates a LoRA adapter from live experience data.

    Args:
        model:             The loaded causal LM (may already be PEFT-wrapped).
        tokenizer:         Associated tokenizer.
        experience_logger: Source of training data.
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> dict:
        """
        Run the full training pipeline synchronously.

        This method blocks and is intended to be called from a background
        thread by the scheduler (so the FastAPI server keeps serving).

        Returns:
            dict with keys: status, episodes_used, train_loss, eval_loss,
                             adapter_path, started_at, completed_at.
        """
        if not _training_lock.acquire(blocking=False):
            logger.warning("Training already in progress. skipping this run.")
            return {"status": "skipped", "reason": "already running"}

        started_at = datetime.utcnow()
        run_record = {
            "run_id": f"run_{started_at.strftime('%Y%m%d_%H%M%S')}",
            "started_at": started_at.isoformat(),
            "status": "started",
            "episodes_used": 0,
            "train_loss": None,
            "eval_loss": None,
            "adapter_path": config.ADAPTER_PATH,
        }
        self._write_run_log(run_record)

        try:
            import asyncio
            loop = asyncio.new_event_loop()
            dataset = loop.run_until_complete(self._builder.build())
            episode_ids = loop.run_until_complete(self._builder.get_episode_ids_for_training())
            loop.close()

            if dataset is None:
                run_record.update({"status": "skipped", "reason": "not_enough_data"})
                self._write_run_log(run_record)
                return run_record

            run_record["episodes_used"] = len(episode_ids)

            metrics = self._run_training(dataset)
            run_record.update(metrics)

            # Mark episodes as used
            loop2 = asyncio.new_event_loop()
            loop2.run_until_complete(self._log.mark_training_used(episode_ids))
            loop2.close()

            run_record["status"] = "completed"
            run_record["completed_at"] = datetime.utcnow().isoformat()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("CUDA OOM during training. See logs.")
                run_record["status"] = "oom"
            else:
                logger.exception("Unexpected RuntimeError during training.")
                run_record["status"] = "failed"
            run_record["error"] = str(e)
        except Exception as e:
            logger.exception("Training failed unexpectedly.")
            run_record["status"] = "failed"
            run_record["error"] = str(e)
        finally:
            _training_lock.release()
            self._write_run_log(run_record)

        return run_record

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_training(self, dataset: DatasetDict, batch_size: Optional[int] = None) -> dict:
        """
        Execute SFTTrainer. On CUDA OOM, halves the batch size and retries once.

        Returns metrics dict with train_loss and eval_loss.
        """
        effective_batch = batch_size or config.BATCH_SIZE

        try:
            return self._do_train(dataset, effective_batch)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and effective_batch > 1:
                halved = max(1, effective_batch // 2)
                logger.warning(
                    "CUDA OOM with batch_size=%d. Retrying with batch_size=%d.",
                    effective_batch,
                    halved,
                )
                gc.collect()
                torch.cuda.empty_cache()
                return self._do_train(dataset, halved)
            raise

    def _do_train(self, dataset: DatasetDict, batch_size: int) -> dict:
        """Inner training call. builds LoRA config and runs SFTTrainer."""
        from peft import PeftModel

        # Properly unload any existing PEFT adapter. Just accessing
        # base_model.base_model.model does NOT remove LoRA layers from
        # submodules, causing get_peft_model() to stack adapters and OOM.
        base_model = self.model
        if isinstance(base_model, PeftModel):
            logger.info("Unloading existing PEFT adapter before training...")
            base_model = base_model.unload()
            self.model = base_model
            gc.collect()
            torch.cuda.empty_cache()

        # Remove leftover peft_config so get_peft_model() doesn't think
        # there's already an adapter attached (causes "modify a model with
        # PEFT for a second time" warning and potential stacking issues).
        if hasattr(base_model, "peft_config"):
            delattr(base_model, "peft_config")

        # Cortex Loop Seam Layer Targeting:
        # Seam layers are where Cortex Loop creates an architectural discontinuity.
        # The model was never trained to accept this layer transition.
        # Higher-rank LoRA here teaches the model to bridge the gap.
        # This is where MIRROR training has the most impact.
        seam_info = self._load_seam_info()
        lora_rank = config.LORA_RANK
        lora_alpha = config.LORA_ALPHA

        if seam_info is not None:
            lora_rank = 32  # Higher rank for seam-targeted training
            lora_alpha = 64
            logger.info(
                "Cortex Loop seam targeting active: rank=%d alpha=%d (seam layers %d→%d)",
                lora_rank, lora_alpha,
                seam_info["seam_entry_layer"], seam_info["seam_exit_layer"],
            )

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=config.LORA_TARGET_MODULES,
            lora_dropout=config.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )

        peft_model = get_peft_model(base_model, lora_config)
        peft_model.print_trainable_parameters()

        adapter_out = Path(config.ADAPTER_PATH)
        adapter_out.mkdir(parents=True, exist_ok=True)

        sft_config = SFTConfig(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=config.GRAD_ACCUM_STEPS,
            warmup_ratio=config.WARMUP_RATIO,
            num_train_epochs=config.NUM_EPOCHS,
            learning_rate=config.LEARNING_RATE,
            fp16=False,
            bf16=True,
            logging_steps=10,
            output_dir=str(adapter_out),
            optim="adamw_8bit",
            save_strategy="epoch",
            eval_strategy="epoch",
            load_best_model_at_end=False,
            report_to="wandb" if config.WANDB_API_KEY else "none",
            run_name=f"prism_lora_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
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
            # On failure (OOM etc), unload the adapter we just applied
            # so the model is clean for a retry. Without this, the next
            # call to get_peft_model() stacks a SECOND adapter on top.
            logger.warning("Training failed. Unloading adapter to prevent stacking...")
            try:
                clean = peft_model.unload()
                self.model = clean
            except Exception:
                pass
            del trainer
            gc.collect()
            torch.cuda.empty_cache()
            raise

        # Save adapter
        trainer.save_model(str(adapter_out))
        logger.info("LoRA adapter saved to %s", adapter_out)

        # Clean up trainer internals (optimizer states, grad buffers)
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

        # Unload adapter from the base model so it's clean for hot-swap.
        # get_peft_model() modifies the base model in-place (injects LoRA layers
        # and adds peft_config). If we leave this dirty state, hot_swap_lora()
        # will stack a second adapter on top, causing Chinese/garbage output.
        clean_base = peft_model.unload()
        if hasattr(clean_base, "peft_config"):
            delattr(clean_base, "peft_config")
        self.model = clean_base
        gc.collect()
        torch.cuda.empty_cache()

        result = {
            "train_loss": train_result.training_loss,
            "eval_loss": eval_result.get("eval_loss"),
        }
        # Include MIRROR delta info if available
        if self._builder._last_avg_delta is not None:
            result["mirror_avg_delta"] = round(self._builder._last_avg_delta, 4)
        return result

    @staticmethod
    def _load_seam_info() -> dict | None:
        """Load Cortex Loop seam layer info if available."""
        seam_path = Path(config.CORTEX_LOOP_SEAM_PATH)
        if not seam_path.exists() or not config.CORTEX_LOOP_ENABLED:
            return None
        try:
            import json as _json
            return _json.loads(seam_path.read_text())
        except Exception:
            return None

    @staticmethod
    def _write_run_log(record: dict) -> None:
        """Append a training run record to data/training_runs.jsonl."""
        _RUNS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _RUNS_LOG.open("a") as f:
            f.write(json.dumps(record) + "\n")
