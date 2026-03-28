"""
PrismInsideClient: synchronous HTTP client for the PRISM server.

Intended for use with Itachi or any Python script that wants to
chat with the PRISM server without dealing with raw HTTP.

Usage:
    from prism_inside_client import PrismInsideClient

    client = PrismInsideClient("http://localhost:8000", session_id="itachi_01")
    result = client.chat("What do you remember about me?")
    print(result["response"])
    client.feedback(result["episode_id"], was_useful=True, score=0.9)
"""

from __future__ import annotations

import json
from typing import Optional

import requests


class PrismInsideClient:
    """
    Synchronous client for the PRISM FastAPI server.

    Args:
        base_url:   Base URL of the server (e.g. "http://localhost:8000").
        session_id: Unique identifier for this conversation session.
        timeout:    HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        session_id: str,
        timeout: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def chat(self, message: str, system_prompt: Optional[str] = None) -> dict:
        """
        Send a message and receive a response.

        Args:
            message:       The user's message.
            system_prompt: Optional system prompt override.

        Returns:
            dict with keys: response (str), episode_id (str), session_id (str).
        """
        payload = {
            "session_id": self.session_id,
            "message": message,
        }
        if system_prompt:
            payload["system_prompt"] = system_prompt

        response = requests.post(
            f"{self.base_url}/chat",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def feedback(
        self,
        episode_id: str,
        was_useful: bool,
        score: float,
        reason: str = "",
    ) -> dict:
        """
        Submit feedback on a response.

        Args:
            episode_id: ID returned by chat().
            was_useful: Whether the response was helpful.
            score:      Quality score 0.0–1.0.
            reason:     Optional explanation.

        Returns:
            dict with keys: episode_id (str), new_fitness (float).
        """
        response = requests.post(
            f"{self.base_url}/feedback",
            json={
                "episode_id": episode_id,
                "was_useful": was_useful,
                "score": score,
                "reason": reason,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def status(self) -> dict:
        """
        Get server health and training status.

        Returns:
            Full StatusResponse dict.
        """
        response = requests.get(
            f"{self.base_url}/status",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def reset_session(self) -> dict:
        """
        Clear the Titans memory state for this session.
        Call this when starting a genuinely new topic.

        Returns:
            dict with keys: status, session_id.
        """
        response = requests.post(
            f"{self.base_url}/admin/reset-session",
            json={"session_id": self.session_id},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def cortex_loop_status(self) -> dict:
        """Get Cortex Loop status."""
        response = requests.get(
            f"{self.base_url}/cortex-loop-status",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def mirror_status(self) -> dict:
        """
        Get MIRROR protocol status and convergence metrics.

        Returns:
            MirrorStatusResponse dict.
        """
        response = requests.get(
            f"{self.base_url}/mirror-status",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def trigger_training(self) -> dict:
        """
        Manually trigger a training run.

        Returns:
            dict with keys: status, episodes_used, training_started_at.
        """
        response = requests.post(
            f"{self.base_url}/admin/trigger-training",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Convenience: interactive chat loop
    # ------------------------------------------------------------------

    def interactive(self) -> None:
        """
        Simple blocking chat loop for terminal use.

        Commands:
            /status          - show server status
            /feedback <score> [reason] - rate last response (0.0-1.0)
            /reset           - clear session memory
            /train           - trigger training manually
            /quit            - exit
        """
        print(f"\n  PRISM Chat  |  session={self.session_id}")
        print("  Commands: /status  /feedback <score>  /mirror  /cortex-loop  /reset  /train  /quit\n")

        last_episode_id: Optional[str] = None

        while True:
            try:
                user_input = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye.")
                break

            if not user_input:
                continue

            if user_input.lower() == "/quit":
                print("Goodbye.")
                break

            if user_input.lower() == "/status":
                s = self.status()
                print(f"\n  Model loaded       : {s['model_loaded']}")
                print(f"  Adapter version    : {s['adapter_version']}")
                print(f"  Total episodes     : {s['total_episodes']}")
                print(f"  High fitness eps   : {s['high_fitness_episodes']}")
                print(f"  Last training run  : {s['last_training_run']}")
                print(f"  GPU mem used/total : {s['gpu_memory_used_gb']:.1f} / {s['gpu_memory_total_gb']:.1f} GB\n")
                continue

            if user_input.lower() == "/cortex-loop":
                try:
                    r = self.cortex_loop_status()
                    print(f"\n  Cortex Loop Status")
                    print(f"  ─────────────────────────────────────")
                    print(f"  Scan complete       : {'Yes' if r['scan_complete'] else 'No'}")
                    if r.get('best_config'):
                        bc = r['best_config']
                        print(f"  Best config found   : (i={bc['i']}, j={bc['j']})")
                        print(f"  Layers duplicated   : {bc['num_layers_duplicated']}")
                        print(f"  Combined delta      : {bc['combined_delta']:+.4f}")
                    print(f"  Model loaded        : {r['model_type']}")
                    if r.get('seam_info'):
                        si = r['seam_info']
                        print(f"  Seam layers         : {si['seam_entry_layer']} → {si['seam_exit_layer']}")
                        print(f"  Total layers        : {si['total_layers_after_cortex_loop']}")
                    print(f"  LoRA seam targeting : {'enabled' if r.get('seam_targeting') else 'disabled'}")
                    print(f"  Scan results        : ./logs/cortex_loop_scan_results.json")
                    print()
                except requests.HTTPError as e:
                    print(f"  Error: {e}\n")
                continue

            if user_input.lower() == "/mirror":
                try:
                    m = self.mirror_status()
                    print(f"\n  MIRROR Protocol Status")
                    print(f"  ──────────────────────────")
                    print(f"  Mode                : {m['mirror_mode']}")
                    print(f"  Oracle calls made   : {m['oracle_calls']}")
                    print(f"  Episodes auto-scored: {m['episodes_auto_scored']}")
                    avg = m['rolling_avg_delta']
                    print(f"  Rolling avg delta   : {avg:.3f}" if avg is not None else "  Rolling avg delta   : N/A")
                    print(f"  Delta sample size   : {m['delta_sample_size']}")
                    print(f"  Convergence thresh  : {m['convergence_threshold']:.2f}")
                    print(f"  Convergence window  : {m['convergence_window']}")
                    est = m.get('estimated_episodes_to_convergence')
                    print(f"  Est. to convergence : {est}" if est is not None else "  Est. to convergence : N/A")
                    print()
                except requests.HTTPError as e:
                    print(f"  Error: {e}\n")
                continue

            if user_input.lower().startswith("/feedback"):
                parts = user_input.split(maxsplit=2)
                if len(parts) < 2 or last_episode_id is None:
                    print("  Usage: /feedback <score 0-1> [reason]\n")
                    continue
                try:
                    score = float(parts[1])
                    reason = parts[2] if len(parts) > 2 else ""
                    r = self.feedback(last_episode_id, was_useful=score >= 0.5, score=score, reason=reason)
                    print(f"  Feedback saved. New fitness: {r['new_fitness']:.3f}\n")
                except (ValueError, requests.HTTPError) as e:
                    print(f"  Error: {e}\n")
                continue

            if user_input.lower() == "/reset":
                self.reset_session()
                print("  Session memory cleared.\n")
                continue

            if user_input.lower() == "/train":
                r = self.trigger_training()
                print(f"  Training: {r['status']} | episodes_used={r['episodes_used']}\n")
                continue

            # Normal chat turn
            try:
                result = self.chat(user_input)
                last_episode_id = result["episode_id"]
                print(f"\n  {result['response']}\n")
            except requests.HTTPError as e:
                print(f"\n  [Error: {e}]\n")


if __name__ == "__main__":
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    session = sys.argv[2] if len(sys.argv) > 2 else "default"
    PrismInsideClient(base, session).interactive()
