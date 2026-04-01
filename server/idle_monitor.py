"""
Idle Monitor: triggers Dream Consolidation during server idle periods.

Inspired by Claude Code's autoDream system that runs memory consolidation
during idle time rather than only on a fixed schedule.

The monitor checks every 60 seconds whether the server has been idle
(no chat messages) for a configurable duration. If idle and cooldown
has elapsed, it queues a Dream Consolidation run.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

import config

logger = logging.getLogger(__name__)


class IdleMonitor:
    """
    Monitors server activity and triggers Dream Consolidation
    during idle periods.

    Args:
        dream_consolidation: The DreamConsolidation instance to trigger.
    """

    def __init__(self, dream_consolidation) -> None:
        self._dream = dream_consolidation
        self._last_activity: datetime = datetime.utcnow()
        self._last_consolidation: Optional[datetime] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._consolidation_in_progress = False

    def record_activity(self) -> None:
        """Record that a chat message was received. Resets the idle timer."""
        self._last_activity = datetime.utcnow()

    def start(self) -> None:
        """Start the idle monitoring loop."""
        if not config.IDLE_CONSOLIDATION_ENABLED:
            logger.info("Idle consolidation disabled.")
            return
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info(
            "IdleMonitor started (idle_minutes=%d, cooldown_minutes=%d).",
            config.IDLE_CONSOLIDATION_MINUTES,
            config.IDLE_CONSOLIDATION_COOLDOWN_MINUTES,
        )

    def stop(self) -> None:
        """Stop the idle monitoring loop."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def _monitor_loop(self) -> None:
        """Check for idle state every 60 seconds."""
        while self._running:
            try:
                await asyncio.sleep(60)
                if self._should_consolidate():
                    await self._run_consolidation()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("IdleMonitor loop error (non-fatal).")

    def _should_consolidate(self) -> bool:
        """Check if conditions are met for idle consolidation."""
        if self._consolidation_in_progress:
            return False

        now = datetime.utcnow()
        idle_duration = now - self._last_activity
        idle_threshold = timedelta(minutes=config.IDLE_CONSOLIDATION_MINUTES)

        if idle_duration < idle_threshold:
            return False

        # Check cooldown
        if self._last_consolidation is not None:
            cooldown = timedelta(minutes=config.IDLE_CONSOLIDATION_COOLDOWN_MINUTES)
            if now - self._last_consolidation < cooldown:
                return False

        # Check if API key is available (consolidation needs Claude)
        if not config.ANTHROPIC_API_KEY:
            return False

        return True

    async def _run_consolidation(self) -> None:
        """Run Dream Consolidation in the background."""
        self._consolidation_in_progress = True
        logger.info("IdleMonitor: server idle for %d+ minutes. Starting Dream Consolidation.",
                     config.IDLE_CONSOLIDATION_MINUTES)
        try:
            report = await self._dream.run()
            self._last_consolidation = datetime.utcnow()
            created = report.get("created", 0) if report else 0
            promoted = report.get("promoted", 0) if report else 0
            logger.info(
                "IdleMonitor: Dream Consolidation complete. created=%d promoted=%d",
                created, promoted,
            )
        except Exception:
            logger.exception("IdleMonitor: Dream Consolidation failed (non-fatal).")
        finally:
            self._consolidation_in_progress = False


# Module-level singleton
_monitor_instance: Optional[IdleMonitor] = None


def get_idle_monitor() -> Optional[IdleMonitor]:
    """Return the global IdleMonitor, or None if not initialized."""
    return _monitor_instance


def set_idle_monitor(monitor: IdleMonitor) -> None:
    """Set the global IdleMonitor instance."""
    global _monitor_instance
    _monitor_instance = monitor
