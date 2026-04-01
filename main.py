"""
PRISM: Entry Point

Startup sequence:
  1. Initialise ExperienceLogger (creates SQLite tables).
  2. Load Qwen2.5-14B-Instruct + adapters via ModelLoader.
  3. Build PrismInferenceEngine.
  4. Build LoRAContinualTrainer + TitansAdapterTrainer.
  5. Build TrainingScheduler and start cron.
  6. Start FastAPI server via uvicorn.

Usage:
    python main.py
    python main.py --port 8080 --no-model  (skip model load for dev)
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import typer
import uvicorn

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

import config
from data.experience_log import get_experience_logger
from data.semantic_store import get_semantic_store
from memory_adapter.memory_state import get_memory_manager
from model.inference import PrismInferenceEngine
from model.loader import ModelLoader
from server.routes import app
from training.contradiction_engine import ContradictionEngine, set_contradiction_engine
from training.dream_consolidation import DreamConsolidation, set_dream_consolidation
from training.lora_trainer import LoRAContinualTrainer
from training.scheduler import TrainingScheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

cli = typer.Typer(add_completion=False)


@cli.command()
def main(
    port: int = typer.Option(config.SERVER_PORT, help="Port to listen on."),
    host: str = typer.Option("0.0.0.0", help="Host to bind to."),
    no_model: bool = typer.Option(
        False, "--no-model", help="Skip model loading (dev mode)."
    ),
    reload: bool = typer.Option(False, help="Enable uvicorn auto-reload (dev only)."),
) -> None:
    """Start the PRISM server."""

    async def _startup() -> None:
        # 1. ExperienceLogger + SemanticStore
        exp_log = get_experience_logger()
        await exp_log.init()
        logger.info("ExperienceLogger ready.")
        # SemanticStore uses the same DB, no separate init needed
        semantic_store = get_semantic_store()
        logger.info("SemanticStore ready.")

        # 2. Model
        loader = ModelLoader()
        if not no_model:
            logger.info("Loading model (this may take a few minutes)...")
            loader.load()
            logger.info("Model loaded.")
        else:
            logger.warning("--no-model flag set: skipping model load (dev mode).")

        # 3. Inference engine
        memory_manager = get_memory_manager()
        engine = PrismInferenceEngine(
            model=loader.model,
            tokenizer=loader.tokenizer,
            memory_manager=memory_manager,
        )

        # 4. Trainers
        lora_trainer = LoRAContinualTrainer(
            model=loader.model,
            tokenizer=loader.tokenizer,
            experience_logger=exp_log,
        )

        # Bootstrap Titans adapter if no checkpoint exists yet.
        # Gate is initialized "closed" (bias=-5) so fresh adapter passes through
        # original embeddings untouched, no corruption. Training opens the gate.
        if not loader.titans_loaded:
            from memory_adapter.titans_adapter import TitansMemoryAdapter
            titans = TitansMemoryAdapter(
                model_hidden_size=loader.model.config.hidden_size,
                memory_size=config.TITANS_MEMORY_SIZE,
                memory_dim=config.TITANS_MEMORY_DIM,
            )
            param = next(loader.model.parameters())
            titans = titans.to(device=param.device, dtype=param.dtype)
            loader.model.titans_adapter = titans
            logger.info("Bootstrapped fresh TitansMemoryAdapter (gate closed).")

        from memory_adapter.train_adapter import TitansAdapterTrainer
        titans_trainer = TitansAdapterTrainer(
            model=loader.model,
            tokenizer=loader.tokenizer,
            experience_logger=exp_log,
        )

        # 5. Dream Consolidation + Contradiction Engine
        dream_consolidation = DreamConsolidation(
            experience_logger=exp_log,
            semantic_store=semantic_store,
        )
        set_dream_consolidation(dream_consolidation)
        logger.info("DreamConsolidation ready.")

        contradiction_engine = ContradictionEngine(experience_logger=exp_log)
        set_contradiction_engine(contradiction_engine)
        logger.info("ContradictionEngine ready.")

        # 6. Active Recall + Scheduler
        active_recall = None
        if config.ACTIVE_RECALL_ENABLED:
            from training.active_recall import ActiveRecallLoop
            active_recall = ActiveRecallLoop(
                model=loader.model,
                tokenizer=loader.tokenizer,
                experience_logger=exp_log,
            )
            logger.info("ActiveRecallLoop ready.")

        scheduler = TrainingScheduler(
            model_loader=loader,
            lora_trainer=lora_trainer,
            titans_trainer=titans_trainer,
            experience_logger=exp_log,
            dream_consolidation=dream_consolidation,
            active_recall=active_recall,
        )
        scheduler.start()

        # 7. Idle Monitor (autoDream-inspired idle-time consolidation)
        if config.IDLE_CONSOLIDATION_ENABLED:
            from server.idle_monitor import IdleMonitor, set_idle_monitor
            idle_monitor = IdleMonitor(dream_consolidation=dream_consolidation)
            set_idle_monitor(idle_monitor)
            # Start after uvicorn event loop is running
            idle_monitor.start()
            logger.info("IdleMonitor ready.")

        # 8. Attach to app.state for route access
        app.state.loader = loader
        app.state.engine = engine
        app.state.scheduler = scheduler

        logger.info(
            "PRISM ready on http://%s:%d",
            host if host != "0.0.0.0" else "localhost",
            port,
        )

    # Run startup before handing off to uvicorn
    asyncio.run(_startup())

    uvicorn.run(
        "server.routes:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    cli()
