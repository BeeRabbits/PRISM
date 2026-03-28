"""
/chat endpoint: runs one inference turn, checks for contradictions, logs the episode.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException

import config
from data.experience_log import get_experience_logger
from data.schemas import ChatRequest, ChatResponse
from model.inference import PrismInferenceEngine
from server.mirror_hook import mirror_post_response
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
      1. Generates a response using PrismInferenceEngine.
      2. Runs ContradictionEngine check on the new episode.
      3. Logs the episode (unless blocked by contradiction).
      4. Returns the response + episode_id.
    """
    exp_log = get_experience_logger()

    try:
        response_text, full_prompt = await engine.generate(
            session_id=request.session_id,
            message=request.message,
            system_prompt=request.system_prompt,
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

    episode_id = await exp_log.log_episode(
        user_message=request.message,
        assistant_response=log_response,
        full_prompt=full_prompt,
        session_id=request.session_id,
    )

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
