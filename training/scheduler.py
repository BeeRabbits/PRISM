"""
TrainingScheduler. APScheduler-based cron for periodic LoRA + Titans training.

Schedule:
  - Default: 3 AM daily (configurable via TRAINING_CRON in .env)
  - Before each run: checks that enough high-fitness episodes exist.
  - After each run: hot-swaps the new LoRA adapter into the live model.
  - Manual trigger: POST /admin/trigger-training calls trigger_now().
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

import config
from data.experience_log import ExperienceLogger, get_experience_logger

logger = logging.getLogger(__name__)


_NIGHTLY_LOG_DIR = Path("logs")


class TrainingScheduler:
    """
    Manages scheduled and on-demand training runs.

    Nightly job order:
      1. Active Recall test (identify forgotten episodes)
      2. ContradictionEngine sweep
      3. DreamConsolidation with TMR (compress + validate + reactivate)
      4. DatasetBuilder.build() (episodes + replay + semantics + general knowledge)
      5. LoRAContinualTrainer.train()
      6. TitansAdapterTrainer.train()
      7. Initialize spaced replay schedules for newly trained episodes
      8. Hot-swap adapters
      9. Write nightly report to logs/nightly_run_{date}.json

    Args:
        model_loader:         ModelLoader instance (for hot-swapping adapters).
        lora_trainer:         LoRAContinualTrainer instance.
        titans_trainer:       TitansAdapterTrainer instance (optional).
        experience_logger:    ExperienceLogger for episode counts.
        dream_consolidation:  DreamConsolidation instance (optional).
        active_recall:        ActiveRecallLoop instance (optional).
    """

    def __init__(
        self,
        model_loader,
        lora_trainer,
        titans_trainer=None,
        experience_logger: Optional[ExperienceLogger] = None,
        dream_consolidation=None,
        active_recall=None,
    ) -> None:
        self._loader = model_loader
        self._lora = lora_trainer
        self._titans = titans_trainer
        self._log = experience_logger or get_experience_logger()
        self._dream = dream_consolidation
        self._recall = active_recall
        self._scheduler = AsyncIOScheduler()
        self._last_run: Optional[datetime] = None
        self._next_run: Optional[datetime] = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the APScheduler and register the cron job."""
        cron_expr = config.TRAINING_CRON  # e.g. "0 3 * * *"
        parts = cron_expr.strip().split()
        if len(parts) != 5:
            logger.error("Invalid TRAINING_CRON: %s. expected 5-field cron. Using '0 3 * * *'.", cron_expr)
            parts = ["0", "3", "*", "*", "*"]

        minute, hour, day, month, day_of_week = parts
        trigger = CronTrigger(
            minute=minute,
            hour=hour,
            day=day,
            month=month,
            day_of_week=day_of_week,
        )

        self._scheduler.add_job(
            self._scheduled_run,
            trigger=trigger,
            id="lora_training",
            name="Nightly LoRA + Titans training",
            replace_existing=True,
        )
        self._scheduler.start()

        # Capture next scheduled run time
        job = self._scheduler.get_job("lora_training")
        if job and job.next_run_time:
            self._next_run = job.next_run_time.replace(tzinfo=None)

        logger.info("TrainingScheduler started. Next run: %s", self._next_run)

    def stop(self) -> None:
        """Shut down the scheduler gracefully."""
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
        logger.info("TrainingScheduler stopped.")

    @property
    def last_run(self) -> Optional[datetime]:
        return self._last_run

    @property
    def next_run(self) -> Optional[datetime]:
        return self._next_run

    # ------------------------------------------------------------------
    # Manual trigger
    # ------------------------------------------------------------------

    async def trigger_now(self) -> dict:
        """
        Manually trigger a training run immediately.
        Called by POST /admin/trigger-training.

        Returns:
            dict with status, episodes_used, training_started_at.
        """
        if self._running:
            return {
                "status": "skipped",
                "reason": "already running",
                "episodes_used": 0,
                "training_started_at": None,
            }

        started_at = datetime.utcnow()
        n = await self._log.count_high_fitness()

        if n < config.TRAINING_MIN_EPISODES:
            logger.info(
                "Manual trigger: not enough episodes (%d / %d).",
                n,
                config.TRAINING_MIN_EPISODES,
            )
            return {
                "status": "skipped",
                "reason": f"only {n} qualifying episodes (need {config.TRAINING_MIN_EPISODES})",
                "episodes_used": n,
                "training_started_at": started_at,
            }

        # Run in background thread so the HTTP response returns immediately
        thread = threading.Thread(target=self._run_training_sync, daemon=True)
        thread.start()

        return {
            "status": "started",
            "episodes_used": n,
            "training_started_at": started_at,
        }

    # ------------------------------------------------------------------
    # Scheduled job
    # ------------------------------------------------------------------

    async def _scheduled_run(self) -> None:
        """Async wrapper called by APScheduler."""
        n = await self._log.count_high_fitness()

        if n < config.TRAINING_MIN_EPISODES:
            logger.info(
                "Scheduled run: not enough data yet: %d episodes, need %d.",
                n,
                config.TRAINING_MIN_EPISODES,
            )
            return

        logger.info("Scheduled training run starting with %d episodes.", n)
        thread = threading.Thread(target=self._run_training_sync, daemon=True)
        thread.start()

    # ------------------------------------------------------------------
    # Core training logic (sync, runs in background thread)
    # ------------------------------------------------------------------

    def _run_training_sync(self) -> None:
        """
        Full nightly pipeline in a background thread:
          1. Active Recall test (identify forgotten episodes)
          2. Dream consolidation with TMR
          3. LoRA training (episodes + replay + semantics + general knowledge)
          4. Titans training
          5. Initialize spaced replay for newly trained episodes
          6. Hot-swap adapters
          7. Write nightly report JSON
        """
        if self._running:
            return
        self._running = True
        self._last_run = datetime.utcnow()

        nightly_report: dict = {
            "started_at": self._last_run.isoformat(),
            "active_recall": None,
            "consolidation": None,
            "lora": None,
            "titans": None,
            "spaced_replay": None,
            "hot_swap": False,
            "completed_at": None,
        }

        try:
            import asyncio

            # Step 0. Active Recall test
            if self._recall is not None and config.ACTIVE_RECALL_ENABLED:
                logger.info("=== Active Recall test start ===")
                loop_recall = asyncio.new_event_loop()
                try:
                    recall_result = loop_recall.run_until_complete(self._recall.run())
                    nightly_report["active_recall"] = recall_result
                    logger.info(
                        "=== Active Recall end: %d/%d weak (avg=%.3f) ===",
                        recall_result.get("weak_count", 0),
                        recall_result.get("total_tested", 0),
                        recall_result.get("avg_score", 0),
                    )
                except Exception:
                    logger.exception("Active Recall failed. continuing to consolidation.")
                finally:
                    loop_recall.close()

            # Step 1. Dream consolidation (now with TMR)
            if self._dream is not None:
                logger.info("=== Dream consolidation (TMR) start ===")
                loop = asyncio.new_event_loop()
                try:
                    consolidation_report = loop.run_until_complete(self._dream.run())
                    nightly_report["consolidation"] = consolidation_report
                    logger.info(
                        "=== Dream consolidation end: %s semantics, %s TMR reactivations ===",
                        consolidation_report.get("semantics_created", 0),
                        consolidation_report.get("tmr_reactivations", 0),
                    )
                except Exception:
                    logger.exception("Dream consolidation failed. continuing to LoRA training.")
                finally:
                    loop.close()

            # Steps 2-3. LoRA + Titans training
            logger.info("=== LoRA training run start ===")
            lora_result = self._lora.train()
            nightly_report["lora"] = lora_result
            logger.info("=== LoRA training run end: %s ===", lora_result.get("status"))

            if lora_result.get("status") == "completed":
                logger.info("Hot-swapping LoRA adapter into live model...")
                self._loader.hot_swap_lora()
                nightly_report["hot_swap"] = True
                logger.info("Hot-swap complete.")

                # Initialize spaced replay for newly trained episodes
                if config.SPACED_REPLAY_ENABLED:
                    try:
                        loop_replay = asyncio.new_event_loop()
                        episode_ids = loop_replay.run_until_complete(
                            self._lora._builder.get_episode_ids_for_training()
                        )
                        loop_replay.run_until_complete(
                            self._lora._builder.init_replay_for_new_episodes(episode_ids)
                        )
                        # Advance replay schedule for replay episodes included in this batch
                        loop_replay.run_until_complete(
                            self._lora._builder.advance_replay_after_training()
                        )
                        loop_replay.close()
                        nightly_report["spaced_replay"] = {
                            "episodes_scheduled": len(episode_ids),
                        }
                        logger.info("Spaced replay: initialized for %d episodes.", len(episode_ids))
                    except Exception:
                        logger.exception("Spaced replay init failed. non-fatal.")

            if self._titans is not None:
                # Free GPU memory from LoRA training before Titans starts
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                logger.info("=== Titans adapter training start ===")
                titans_result = self._titans.train()
                nightly_report["titans"] = titans_result
                logger.info("=== Titans training end: %s ===", titans_result.get("status"))

            # MIRROR convergence check
            try:
                loop_mirror = asyncio.new_event_loop()
                mirror_stats = loop_mirror.run_until_complete(self._log.get_mirror_stats())
                loop_mirror.close()
                nightly_report["mirror"] = mirror_stats
                avg_delta = mirror_stats.get("rolling_avg_delta")
                sample_size = mirror_stats.get("delta_sample_size", 0)
                if (
                    avg_delta is not None
                    and sample_size >= config.MIRROR_CONVERGENCE_WINDOW
                    and avg_delta < config.MIRROR_CONVERGENCE_THRESHOLD
                    and not config.MIRROR_CONVERGED
                ):
                    logger.info(
                        "MIRROR CONVERGED: avg_delta=%.3f < threshold=%.3f over %d episodes. "
                        "Oracle scoring will be disabled.",
                        avg_delta, config.MIRROR_CONVERGENCE_THRESHOLD, sample_size,
                    )
                    config.MIRROR_CONVERGED = True
                    nightly_report["mirror_converged"] = True
            except Exception:
                logger.exception("MIRROR convergence check failed. non-fatal.")

        except Exception:
            logger.exception("Unexpected error in background training run.")
        finally:
            self._running = False
            nightly_report["completed_at"] = datetime.utcnow().isoformat()
            self._write_nightly_report(nightly_report)
            job = self._scheduler.get_job("lora_training")
            if job and job.next_run_time:
                self._next_run = job.next_run_time.replace(tzinfo=None)

    @staticmethod
    def _write_nightly_report(report: dict) -> None:
        """Write the nightly run report to logs/nightly_run_{date}.json."""
        _NIGHTLY_LOG_DIR.mkdir(parents=True, exist_ok=True)
        date_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = _NIGHTLY_LOG_DIR / f"nightly_run_{date_str}.json"
        try:
            path.write_text(json.dumps(report, indent=2, default=str))
            logger.info("Nightly report written to %s", path)
        except Exception as e:
            logger.error("Failed to write nightly report: %s", e)
