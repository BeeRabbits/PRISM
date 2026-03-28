"""Lazy Anthropic async client singleton."""

from __future__ import annotations

import logging
from typing import Any, Optional

import config

logger = logging.getLogger(__name__)

_client: Optional[Any] = None


def get_async_anthropic_client() -> Optional[Any]:
    """
    Return a cached AsyncAnthropic client, creating it lazily on first call.

    Uses config.ANTHROPIC_API_KEY. Returns None if the key is missing or
    the anthropic package is not installed.
    """
    global _client
    if _client is not None:
        return _client
    if not config.ANTHROPIC_API_KEY:
        logger.warning(
            "ANTHROPIC_API_KEY not set. Anthropic client unavailable."
        )
        return None
    try:
        import anthropic
        _client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
        return _client
    except ImportError:
        logger.error(
            "'anthropic' package not installed. Run: pip install anthropic"
        )
        return None
