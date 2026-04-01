"""
/chat endpoint: runs one inference turn, checks for contradictions, logs the episode.

Integrations:
  - Frustration Detection: regex-based sentiment before inference
  - Idle Monitor: records activity for idle-time consolidation
  - Session History: server-side conversation tracking
  - Context Compaction: summarize long sessions to stay focused
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException

import config
from data.experience_log import get_experience_logger
from data.schemas import ChatRequest, ChatResponse
from model.inference import PrismInferenceEngine
from server.frustration_detector import detect_frustration
from server.mirror_hook import mirror_post_response
from server.session_history import get_session_history
from training.contradiction_engine import get_contradiction_engine

logger = logging.getLogger(__name__)

router = APIRouter()


async def chat_endpoint(
    request: ChatRequest,
    engine: PrismInferenceEngine,
) -> ChatResponse:
    """
    POST /chat

    Runs one inference turn:
      1. Detects user frustration (regex, zero API cost).
      2. Generates a response using PrismInferenceEngine.
      3. Runs ContradictionEngine check on the new episode.
      4. Logs the episode with sentiment metadata.
      5. Returns the response + episode_id.
    """
    exp_log = get_experience_logger()

    # Record activity for idle monitor
    from server.idle_monitor import get_idle_monitor
    idle_monitor = get_idle_monitor()
    if idle_monitor is not None:
        idle_monitor.record_activity()

    # Frustration detection (pre-inference, zero API cost)
    frustration = None
    effective_system_prompt = request.system_prompt
    if config.FRUSTRATION_DETECTION_ENABLED:
        frustration = detect_frustration(request.message)
        if frustration.tier.value > 0 and frustration.recommended_modifier:
            base = request.system_prompt or ""
            effective_system_prompt = (base + "\n" + frustration.recommended_modifier).strip() or None

    # Session history: track conversation for context compaction
    session_hist = get_session_history()
    session_hist.add_turn(request.session_id, "user", request.message)
    conversation_history = session_hist.get_inference_history(request.session_id) if config.COMPACTION_ENABLED else None

    try:
        response_text, full_prompt = await engine.generate(
            session_id=request.session_id,
            message=request.message,
            system_prompt=effective_system_prompt,
            conversation_history=conversation_history,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("CUDA OOM during inference for session %s", request.session_id)
            raise HTTPException(status_code=503, detail="GPU out of memory. Try a shorter message.")
        logger.exception("Inference error for session %s", request.session_id)
        raise HTTPException(status_code=500, detail=str(e))

    # --- Contradiction check ---
    contradiction_engine = get_contradiction_engine()
    episode_content = f"User: {request.message}\nAssistant: {response_text}"
    final_content = episode_content
    episode_id = ""

    if contradiction_engine is not None:
        try:
            c_result = await contradiction_engine.check_and_resolve(
                new_episode_content=episode_content,
                user_id=request.session_id,
            )
            if c_result.final_action == "blocked":
                # Contradiction won. Don't log this episode, but still return the response
                logger.info(
                    "ContradictionEngine blocked episode for session %s",
                    request.session_id,
                )
                return ChatResponse(
                    response=response_text,
                    episode_id="",
                    session_id=request.session_id,
                )
            elif c_result.final_action == "merged" and c_result.final_content:
                final_content = c_result.final_content
        except Exception as e:
            logger.error("ContradictionEngine check failed. Storing episode normally: %s", e)

    # Log the episode (original or merged content)
    # Parse merged content back to user/assistant if it was merged
    log_response = response_text
    if final_content != episode_content and "\nAssistant: " in final_content:
        log_response = final_content.split("\nAssistant: ", 1)[-1]

    # Build sentiment metadata for episode logging
    sentiment_tier = None
    sentiment_patterns = None
    if frustration is not None and frustration.tier.value > 0:
        sentiment_tier = frustration.tier.name
        sentiment_patterns = json.dumps(frustration.matched_patterns)

    episode_id = await exp_log.log_episode(
        user_message=request.message,
        assistant_response=log_response,
        full_prompt=full_prompt,
        session_id=request.session_id,
        sentiment_tier=sentiment_tier,
        sentiment_patterns=sentiment_patterns,
    )

    # Track assistant response in session history
    session_hist.add_turn(request.session_id, "assistant", response_text)

    # Background compaction if session is getting long
    if config.COMPACTION_ENABLED and session_hist.needs_compaction(request.session_id):
        from server.compaction_service import get_compaction_service
        compaction = get_compaction_service()
        asyncio.create_task(compaction.compact_if_needed(request.session_id))

    # MIRROR: fire background scoring after response is ready
    if config.MIRROR_ENABLED and episode_id:
        asyncio.create_task(
            mirror_post_response(
                episode_id=episode_id,
                user_message=request.message,
                prism_response=response_text,
                model=engine.model,
                tokenizer=engine.tokenizer,
            )
        )

    return ChatResponse(
        response=response_text,
        episode_id=episode_id,
        session_id=request.session_id,
    )
