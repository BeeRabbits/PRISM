"""
/status endpoint: returns live system health and training stats.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import torch

from data.experience_log import get_experience_logger
from data.schemas import StatusResponse
from model.loader import ModelLoader
from training.scheduler import TrainingScheduler

logger = logging.getLogger(__name__)


async def status_endpoint(
    loader: ModelLoader,
    scheduler: TrainingScheduler,
) -> StatusResponse:
    """
    GET /status

    Returns a snapshot of:
      - Model readiness
      - Adapter versions
      - Episode counts
      - Training schedule
      - GPU memory
    """
    exp_log = get_experience_logger()

    total = await exp_log.count_total()
    high = await exp_log.count_high_fitness()

    # GPU memory
    cuda_available = torch.cuda.is_available()
    gpu_used_gb = 0.0
    gpu_total_gb = 0.0
    if cuda_available:
        try:
            props = torch.cuda.get_device_properties(0)
            gpu_total_gb = props.total_memory / 1e9
            gpu_used_gb = (
                props.total_memory - torch.cuda.mem_get_info(0)[0]
            ) / 1e9
        except Exception:
            pass

    return StatusResponse(
        model_loaded=loader.is_ready,
        adapter_version=loader.adapter_version,
        titans_adapter_loaded=loader.titans_loaded,
        total_episodes=total,
        high_fitness_episodes=high,
        last_training_run=scheduler.last_run,
        next_training_run=scheduler.next_run,
        cuda_available=cuda_available,
        gpu_memory_used_gb=round(gpu_used_gb, 2),
        gpu_memory_total_gb=round(gpu_total_gb, 2),
    )
