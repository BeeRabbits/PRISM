"""
Pydantic models for PRISM data layer.

These are used for:
  - API request/response validation (FastAPI)
  - Internal data transfer between components
  - SQLAlchemy ORM ↔ Python object bridging
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Episode (a single interaction turn)
# ---------------------------------------------------------------------------

class EpisodeCreate(BaseModel):
    """Input when logging a new episode."""
    session_id: str
    user_message: str
    assistant_response: str
    full_prompt: str
    memory_context_used: str = ""


class EpisodeOut(BaseModel):
    """Episode returned from the database."""
    id: uuid.UUID
    session_id: str
    user_message: str
    assistant_response: str
    full_prompt: str
    memory_context_used: str
    response_quality: Optional[float]
    was_memory_useful: Optional[bool]
    training_used: bool
    fitness_score: float
    timestamp: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Fitness signal
# ---------------------------------------------------------------------------

class FitnessSignalCreate(BaseModel):
    """Input when submitting a feedback/fitness signal."""
    episode_id: uuid.UUID
    signal_type: str = Field(
        ..., pattern="^(explicit|implicit|contradiction)$"
    )
    score: float = Field(..., ge=0.0, le=1.0)
    reason: str = ""


class FitnessSignalOut(BaseModel):
    id: uuid.UUID
    episode_id: uuid.UUID
    signal_type: str
    score: float
    reason: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Training run metadata
# ---------------------------------------------------------------------------

class TrainingRunRecord(BaseModel):
    """Logged to data/training_runs.jsonl after each training run."""
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    started_at: datetime
    completed_at: Optional[datetime] = None
    episodes_used: int
    train_loss: Optional[float] = None
    eval_loss: Optional[float] = None
    adapter_path: str
    status: str = "started"  # started | completed | failed | oom


# ---------------------------------------------------------------------------
# API request/response schemas (used by FastAPI routes)
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str
    message: str
    system_prompt: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    episode_id: str
    session_id: str


class FeedbackRequest(BaseModel):
    episode_id: str
    was_useful: bool
    score: float = Field(..., ge=0.0, le=1.0)
    reason: str = ""


class FeedbackResponse(BaseModel):
    episode_id: str
    new_fitness: float


class StatusResponse(BaseModel):
    model_loaded: bool
    adapter_version: str
    titans_adapter_loaded: bool
    total_episodes: int
    high_fitness_episodes: int
    last_training_run: Optional[datetime]
    next_training_run: Optional[datetime]
    cuda_available: bool
    gpu_memory_used_gb: float
    gpu_memory_total_gb: float


class TriggerTrainingResponse(BaseModel):
    status: str
    episodes_used: int
    training_started_at: datetime


class ResetSessionRequest(BaseModel):
    session_id: str


class ResetSessionResponse(BaseModel):
    status: str
    session_id: str


# ---------------------------------------------------------------------------
# Semantic memory schemas
# ---------------------------------------------------------------------------

class SemanticMemoryOut(BaseModel):
    id: str
    content: str
    source_episode_ids: list
    key_pattern: str
    confidence: float
    is_provisional: bool
    validation_count: int
    used_in_training: bool
    fitness_score: float
    created_at: datetime
    promoted_at: Optional[datetime]


class SemanticsListResponse(BaseModel):
    confirmed: list[SemanticMemoryOut]
    provisional: list[SemanticMemoryOut]


# ---------------------------------------------------------------------------
# Contradiction log schemas
# ---------------------------------------------------------------------------

class ContradictionLogOut(BaseModel):
    id: str
    episode_a_id: str
    episode_b_id: Optional[str]
    resolution: str
    reason: str
    confidence: float
    resolved_at: datetime


class ContradictionListResponse(BaseModel):
    contradictions: list[ContradictionLogOut]


# ---------------------------------------------------------------------------
# Dream consolidation schemas
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# MIRROR Protocol schemas
# ---------------------------------------------------------------------------

class MirrorStatusResponse(BaseModel):
    mirror_mode: str  # active | converged | disabled
    oracle_calls: int
    rolling_avg_delta: Optional[float]
    episodes_auto_scored: int
    convergence_threshold: float
    convergence_window: int
    delta_sample_size: int
    estimated_episodes_to_convergence: Optional[int]


class ConsolidationReportResponse(BaseModel):
    clusters_found: int
    clusters_skipped_low_confidence: int
    clusters_skipped_too_varied: int
    semantics_created: int
    semantics_promoted: int
    semantics_pruned: int
    duration_seconds: float
    ran_at: Optional[str]
