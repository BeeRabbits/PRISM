"""
FastAPI application assembly: wires together all routes.

The app object here is imported by main.py.

Dependency injection pattern:
  - ModelLoader, PrismInferenceEngine, TrainingScheduler are
    instantiated in main.py and injected into routes via
    app.state, accessed through request.app.state.
"""

from __future__ import annotations

import logging
from datetime import datetime

from fastapi import FastAPI, Request

from data.schemas import (
    ChatRequest,
    ChatResponse,
    ConsolidationReportResponse,
    ContradictionListResponse,
    FeedbackRequest,
    FeedbackResponse,
    MirrorStatusResponse,
    ResetSessionRequest,
    ResetSessionResponse,
    SemanticMemoryOut,
    SemanticsListResponse,
    StatusResponse,
    TriggerTrainingResponse,
)
from data.experience_log import get_experience_logger
from data.semantic_store import get_semantic_store
from memory_adapter.memory_state import get_memory_manager
from server.chat import chat_endpoint
from server.feedback import feedback_endpoint
from server.status import status_endpoint
from training.contradiction_engine import get_contradiction_engine
from training.dream_consolidation import get_dream_consolidation, get_last_consolidation_report

logger = logging.getLogger(__name__)

app = FastAPI(
    title="PRISM",
    description=(
        "Persistent Resonance Intelligence & Self-Modifying Memory. "
        "Qwen2.5-14B with QLoRA continual learning and Titans memory adapter."
    ),
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    """Run one inference turn."""
    engine = request.app.state.engine
    return await chat_endpoint(body, engine)


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(body: FeedbackRequest) -> FeedbackResponse:
    """Submit a fitness signal for a logged episode."""
    return await feedback_endpoint(body)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@app.get("/status", response_model=StatusResponse)
async def status(request: Request) -> StatusResponse:
    """Return live system health and training stats."""
    loader = request.app.state.loader
    scheduler = request.app.state.scheduler
    return await status_endpoint(loader, scheduler)


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------

@app.post("/admin/trigger-training", response_model=TriggerTrainingResponse)
async def trigger_training(request: Request) -> TriggerTrainingResponse:
    """Manually trigger a training run immediately."""
    scheduler = request.app.state.scheduler
    result = await scheduler.trigger_now()
    return TriggerTrainingResponse(
        status=result["status"],
        episodes_used=result.get("episodes_used", 0),
        training_started_at=result.get("training_started_at") or datetime.utcnow(),
    )


@app.post("/admin/reset-session", response_model=ResetSessionResponse)
async def reset_session(body: ResetSessionRequest) -> ResetSessionResponse:
    """Clear the Titans memory state for a session."""
    memory_manager = get_memory_manager()
    await memory_manager.reset(body.session_id)
    return ResetSessionResponse(
        status="cleared",
        session_id=body.session_id,
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Semantic memories
# ---------------------------------------------------------------------------

@app.get("/semantics", response_model=SemanticsListResponse)
async def get_semantics() -> SemanticsListResponse:
    """Return all confirmed and provisional semantic memories."""
    store = get_semantic_store()
    confirmed_raw = await store.get_confirmed_semantics(limit=200)
    provisional_raw = await store.get_provisional_semantics()

    def _to_out(s: dict) -> SemanticMemoryOut:
        return SemanticMemoryOut(
            id=s["id"],
            content=s["content"],
            source_episode_ids=s["source_episode_ids"],
            key_pattern=s["key_pattern"],
            confidence=s["confidence"],
            is_provisional=s["is_provisional"],
            validation_count=s["validation_count"],
            used_in_training=s["used_in_training"],
            fitness_score=s["fitness_score"],
            created_at=s["created_at"],
            promoted_at=s.get("promoted_at"),
        )

    return SemanticsListResponse(
        confirmed=[_to_out(s) for s in confirmed_raw],
        provisional=[_to_out(s) for s in provisional_raw],
    )


@app.get("/semantics/{semantic_id}", response_model=SemanticMemoryOut)
async def get_semantic(semantic_id: str) -> SemanticMemoryOut:
    """Return a single semantic memory by ID."""
    from fastapi import HTTPException
    store = get_semantic_store()
    s = await store.get_by_id(semantic_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Semantic memory not found.")
    return SemanticMemoryOut(
        id=s["id"],
        content=s["content"],
        source_episode_ids=s["source_episode_ids"],
        key_pattern=s["key_pattern"],
        confidence=s["confidence"],
        is_provisional=s["is_provisional"],
        validation_count=s["validation_count"],
        used_in_training=s["used_in_training"],
        fitness_score=s["fitness_score"],
        created_at=s["created_at"],
        promoted_at=s.get("promoted_at"),
    )


@app.delete("/semantics/{semantic_id}")
async def delete_semantic(semantic_id: str) -> dict:
    """Hard-delete a semantic memory (admin use)."""
    store = get_semantic_store()
    await store.delete_semantic(semantic_id)
    return {"status": "deleted", "id": semantic_id}


# ---------------------------------------------------------------------------
# Contradiction log
# ---------------------------------------------------------------------------

@app.get("/contradictions", response_model=ContradictionListResponse)
async def get_contradictions(limit: int = 50, offset: int = 0) -> ContradictionListResponse:
    """Return recent contradiction log entries."""
    from data.schemas import ContradictionLogOut
    exp_log = get_experience_logger()
    rows = await exp_log.get_contradictions(limit=limit, offset=offset)
    return ContradictionListResponse(
        contradictions=[
            ContradictionLogOut(
                id=r["id"],
                episode_a_id=r["episode_a_id"],
                episode_b_id=r.get("episode_b_id"),
                resolution=r["resolution"],
                reason=r["reason"],
                confidence=r["confidence"],
                resolved_at=r["resolved_at"],
            )
            for r in rows
        ]
    )


# ---------------------------------------------------------------------------
# Cortex Loop
# ---------------------------------------------------------------------------

@app.get("/cortex-loop-status")
async def cortex_loop_status(request: Request) -> dict:
    """Return Cortex Loop status."""
    import config as _cfg
    from pathlib import Path
    import json as _json

    loader = request.app.state.loader

    result: dict = {
        "enabled": _cfg.CORTEX_LOOP_ENABLED,
        "scan_complete": _cfg.CORTEX_LOOP_SCAN_COMPLETE,
        "model_type": "cortex-loop-enhanced" if loader.cortex_loop_active else "base",
        "best_config": None,
        "seam_info": None,
        "seam_targeting": False,
    }

    config_path = Path(_cfg.CORTEX_LOOP_CONFIG_PATH)
    if config_path.exists():
        try:
            result["best_config"] = _json.loads(config_path.read_text())
        except Exception:
            pass

    seam_path = Path(_cfg.CORTEX_LOOP_SEAM_PATH)
    if seam_path.exists():
        try:
            result["seam_info"] = _json.loads(seam_path.read_text())
            result["seam_targeting"] = _cfg.CORTEX_LOOP_ENABLED
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# MIRROR Protocol
# ---------------------------------------------------------------------------

@app.get("/mirror-status", response_model=MirrorStatusResponse)
async def mirror_status() -> MirrorStatusResponse:
    """Return MIRROR protocol status and convergence metrics."""
    import config as _cfg
    exp_log = get_experience_logger()
    stats = await exp_log.get_mirror_stats()

    if not _cfg.MIRROR_ENABLED:
        mode = "disabled"
    elif _cfg.MIRROR_CONVERGED:
        mode = "converged"
    else:
        mode = "active"

    avg = stats["rolling_avg_delta"]
    threshold = _cfg.MIRROR_CONVERGENCE_THRESHOLD
    window = _cfg.MIRROR_CONVERGENCE_WINDOW
    sample = stats["delta_sample_size"]

    # Rough estimate: if delta is decreasing, how many more episodes needed
    est = None
    if avg is not None and avg > threshold and sample >= 5:
        # Simple linear extrapolation, will be refined over time
        gap = avg - threshold
        est = int(gap / 0.01) if gap > 0 else 0  # ~1 episode per 0.01 delta reduction

    return MirrorStatusResponse(
        mirror_mode=mode,
        oracle_calls=stats["oracle_calls"],
        rolling_avg_delta=avg,
        episodes_auto_scored=stats["auto_scored_count"],
        convergence_threshold=threshold,
        convergence_window=window,
        delta_sample_size=sample,
        estimated_episodes_to_convergence=est,
    )


# ---------------------------------------------------------------------------
# Dream consolidation admin
# ---------------------------------------------------------------------------

@app.get("/admin/consolidation-report", response_model=ConsolidationReportResponse)
async def consolidation_report() -> ConsolidationReportResponse:
    """Return the last dream consolidation run report."""
    from fastapi import HTTPException
    report = get_last_consolidation_report()
    if report is None:
        raise HTTPException(status_code=404, detail="No consolidation run has completed yet.")
    return ConsolidationReportResponse(**report)


@app.post("/admin/run-consolidation", response_model=ConsolidationReportResponse)
async def run_consolidation() -> ConsolidationReportResponse:
    """Manually trigger dream consolidation now."""
    from fastapi import HTTPException
    consolidation = get_dream_consolidation()
    if consolidation is None:
        raise HTTPException(status_code=503, detail="DreamConsolidation not initialised.")
    report = await consolidation.run()
    return ConsolidationReportResponse(**report)
