"""
EvalRunner: automated evaluation gate for training runs.

Runs the test suite against the model, scores responses using both
heuristic checks and optional oracle scoring, compares against stored
baselines, and returns a pass/fail verdict.

Integration:
    Called by TrainingScheduler AFTER training but BEFORE hot-swap.
    If the evaluation fails, the adapter is rolled back and not hot-swapped.

Neuroscience parallel:
    This is the "quality control" mechanism that the brain uses during
    consolidation — not all replayed memories get consolidated into
    long-term storage. Memories that would disrupt existing knowledge
    are suppressed (prefrontal cortical ripples suppress hippocampal
    reactivation during sleep — Current Biology, 2024).
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

import config
from evaluation.test_suite import ALL_TESTS, GENERAL_TESTS, PERSONAL_TESTS, TestCase

logger = logging.getLogger(__name__)

_BASELINE_PATH = Path("data/eval_baseline.json")
_EVAL_LOG_DIR = Path("logs/evaluations")


# ---------------------------------------------------------------------------
# Heuristic checks (zero API cost)
# ---------------------------------------------------------------------------

def _check_has_structure(response: str) -> bool:
    """Response has markdown formatting (headers, bullets, numbered lists)."""
    patterns = [r"\*\*", r"^[-*] ", r"^\d+\.", r"^#{1,3} "]
    return any(re.search(p, response, re.MULTILINE) for p in patterns)


def _check_no_repetition(response: str) -> bool:
    """No phrase repeated 3+ times consecutively."""
    words = response.split()
    if len(words) < 10:
        return True
    # Check for repeated 3-grams
    for i in range(len(words) - 9):
        trigram = " ".join(words[i:i+3])
        count = 0
        for j in range(i, min(i + 30, len(words) - 2)):
            if " ".join(words[j:j+3]) == trigram:
                count += 1
        if count >= 3:
            return False
    return True


def _check_no_gibberish(response: str) -> bool:
    """No obvious degenerate output (word salad, token soup)."""
    if not response.strip():
        return False
    # Check for very long "words" (token soup)
    words = response.split()
    long_words = sum(1 for w in words if len(w) > 30)
    if long_words > 3:
        return False
    # Check for excessive punctuation ratio
    punct_count = sum(1 for c in response if c in "_.!?;:,")
    if len(response) > 0 and punct_count / len(response) > 0.3:
        return False
    return True


def _check_no_chinese(response: str) -> bool:
    """No significant Chinese character content (>5% of response)."""
    chinese_count = sum(1 for c in response if '\u4e00' <= c <= '\u9fff')
    if len(response) > 0 and chinese_count / len(response) > 0.05:
        return False
    return True


def _check_min_length(response: str, min_len: int) -> bool:
    """Response meets minimum character length."""
    return len(response.strip()) >= min_len


def _check_contains(response: str, keyword: str) -> bool:
    """Response contains keyword (case-insensitive)."""
    return keyword.lower() in response.lower()


def run_check(check_name: str, response: str) -> bool:
    """Run a named check against a response string."""
    if check_name == "has_structure":
        return _check_has_structure(response)
    elif check_name == "no_repetition":
        return _check_no_repetition(response)
    elif check_name == "no_gibberish":
        return _check_no_gibberish(response)
    elif check_name == "no_chinese":
        return _check_no_chinese(response)
    elif check_name.startswith("min_length_"):
        min_len = int(check_name.split("_")[-1])
        return _check_min_length(response, min_len)
    elif check_name == "contains_landen":
        return _check_contains(response, "landen")
    elif check_name == "contains_aws":
        return _check_contains(response, "aws")
    elif check_name == "contains_wgu":
        return _check_contains(response, "wgu") or _check_contains(response, "western governors")
    elif check_name == "contains_prism":
        return _check_contains(response, "prism")
    elif check_name == "contains_2pm_or_afternoon":
        return _check_contains(response, "2") or _check_contains(response, "afternoon")
    elif check_name == "contains_sam":
        return _check_contains(response, "sam")
    elif check_name == "contains_ai_ml":
        return (_check_contains(response, "ai") or _check_contains(response, "machine learning")
                or _check_contains(response, "artificial intelligence"))
    elif check_name == "contains_archangel_or_michael":
        return _check_contains(response, "archangel") or _check_contains(response, "michael")
    elif check_name == "contains_april":
        return _check_contains(response, "april") or _check_contains(response, "13")
    elif check_name == "contains_solo_leveling":
        return _check_contains(response, "solo leveling")
    else:
        logger.warning("Unknown check: %s", check_name)
        return True


# ---------------------------------------------------------------------------
# EvalRunner
# ---------------------------------------------------------------------------

class EvalRunner:
    """
    Runs the evaluation test suite and produces a pass/fail verdict.

    Args:
        model:     The model to evaluate.
        tokenizer: The associated tokenizer.
    """

    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    async def run_evaluation(
        self,
        test_cases: Optional[List[TestCase]] = None,
        label: str = "post_training",
    ) -> dict:
        """
        Run all test cases and return results.

        Returns:
            dict with keys:
                - label: evaluation label
                - timestamp: when the eval ran
                - results: list of per-test results
                - general_score: fraction of general tests passed
                - personal_score: fraction of personal tests passed
                - overall_pass: whether the evaluation gate passes
                - duration_seconds: how long it took
        """
        cases = test_cases or ALL_TESTS
        t0 = time.monotonic()

        results = []
        for tc in cases:
            response = await self._generate_response(tc.prompt)
            checks_passed = {}
            for check in tc.checks:
                checks_passed[check] = run_check(check, response)

            all_passed = all(checks_passed.values()) if checks_passed else True
            results.append({
                "id": tc.id,
                "category": tc.category,
                "prompt": tc.prompt,
                "response_preview": response[:200],
                "response_length": len(response),
                "checks": checks_passed,
                "passed": all_passed,
            })

        # Score by category
        general_results = [r for r in results if r["category"] == "general"]
        personal_results = [r for r in results if r["category"] == "personal"]

        general_passed = sum(1 for r in general_results if r["passed"])
        personal_passed = sum(1 for r in personal_results if r["passed"])

        general_score = general_passed / max(1, len(general_results))
        personal_score = personal_passed / max(1, len(personal_results))

        # Gate logic: general MUST be >= 80%, personal is informational
        overall_pass = general_score >= 0.80

        duration = round(time.monotonic() - t0, 2)

        report = {
            "label": label,
            "timestamp": datetime.utcnow().isoformat(),
            "results": results,
            "general_score": round(general_score, 3),
            "general_passed": general_passed,
            "general_total": len(general_results),
            "personal_score": round(personal_score, 3),
            "personal_passed": personal_passed,
            "personal_total": len(personal_results),
            "overall_pass": overall_pass,
            "duration_seconds": duration,
        }

        # Log the report
        self._write_eval_log(report)

        logger.info(
            "Eval [%s]: general=%d/%d (%.0f%%) personal=%d/%d (%.0f%%) → %s (%.1fs)",
            label,
            general_passed, len(general_results), general_score * 100,
            personal_passed, len(personal_results), personal_score * 100,
            "PASS" if overall_pass else "FAIL",
            duration,
        )

        return report

    async def capture_baseline(self) -> dict:
        """
        Run evaluation and save as the baseline.
        Call this when the model is in a known-good state (e.g., base model, no LoRA).
        """
        report = await self.run_evaluation(label="baseline")
        _BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _BASELINE_PATH.write_text(json.dumps(report, indent=2, default=str))
        logger.info("Baseline captured: general=%.0f%% personal=%.0f%%",
                     report["general_score"] * 100, report["personal_score"] * 100)
        return report

    def compare_to_baseline(self, current: dict) -> dict:
        """
        Compare current evaluation against stored baseline.

        Returns:
            dict with general_delta, personal_delta, and regression flag.
        """
        if not _BASELINE_PATH.exists():
            logger.warning("No baseline found at %s. Skipping comparison.", _BASELINE_PATH)
            return {"has_baseline": False}

        baseline = json.loads(_BASELINE_PATH.read_text())
        general_delta = current["general_score"] - baseline["general_score"]
        personal_delta = current["personal_score"] - baseline["personal_score"]

        # Regression: general capability dropped by more than 10%
        regression = general_delta < -0.10

        result = {
            "has_baseline": True,
            "baseline_general": baseline["general_score"],
            "baseline_personal": baseline["personal_score"],
            "current_general": current["general_score"],
            "current_personal": current["personal_score"],
            "general_delta": round(general_delta, 3),
            "personal_delta": round(personal_delta, 3),
            "regression_detected": regression,
        }

        if regression:
            logger.warning(
                "REGRESSION DETECTED: general %.0f%% → %.0f%% (delta=%.1f%%)",
                baseline["general_score"] * 100,
                current["general_score"] * 100,
                general_delta * 100,
            )
        else:
            logger.info(
                "Eval comparison: general %.0f%%→%.0f%% (Δ%.1f%%) personal %.0f%%→%.0f%% (Δ%.1f%%)",
                baseline["general_score"] * 100, current["general_score"] * 100, general_delta * 100,
                baseline["personal_score"] * 100, current["personal_score"] * 100, personal_delta * 100,
            )

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _generate_response(self, prompt: str) -> str:
        """Generate a response for a test prompt (direct model call, no memory injection)."""
        system = "You are PRISM, a persistent intelligent assistant."
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        device = next(self.model.parameters()).device
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if hasattr(tokenized, "input_ids"):
            input_ids = tokenized.input_ids.to(device)
        else:
            input_ids = tokenized.to(device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_ids.shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return response

    @staticmethod
    def _write_eval_log(report: dict) -> None:
        """Write evaluation report to logs/evaluations/."""
        _EVAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = _EVAL_LOG_DIR / f"eval_{report['label']}_{ts}.json"
        try:
            path.write_text(json.dumps(report, indent=2, default=str))
        except Exception as e:
            logger.error("Failed to write eval log: %s", e)
