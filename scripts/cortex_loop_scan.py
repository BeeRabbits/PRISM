# Cortex Loop: layer duplication for enhanced reasoning depth.
# Inspired by the RYS technique: https://github.com/dnhkng/RYS
"""
Cortex Loop Scanner: sweep (i, j) layer duplication configs to find the
optimal reasoning circuit in Qwen2.5-7B.

Usage:
    python scripts/cortex_loop_scan.py               # Full sweep (~378 configs)
    python scripts/cortex_loop_scan.py --fast         # Block sizes 2-8 only (~168 configs)
    python scripts/cortex_loop_scan.py --fast --resume  # Resume interrupted scan
    python scripts/cortex_loop_scan.py --fast --mode memory   # Memory cortex scan only
    python scripts/cortex_loop_scan.py --fast --mode full     # All three probe sets

Based on David Noel Ng's RYS research: https://dnhkng.github.io/posts/rys
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
import config  # noqa: E402

# ---------------------------------------------------------------------------
# Math probes (10 questions with known answers)
# ---------------------------------------------------------------------------

MATH_PROBES = [
    ("What is the cube root of 19034163? Answer with only the integer.", 267),
    ("What is 8741 multiplied by 9372? Answer with only the integer.", 81920652),
    ("What is the square root of 7485321? Answer with only the integer.", 2736),
    ("What is 472 to the power of 3? Answer with only the integer.", 105154048),
    ("What is the cube root of 857375? Answer with only the integer.", 95),
    ("What is 6237 multiplied by 4891? Answer with only the integer.", 30505167),
    ("What is the square root of 2809? Answer with only the integer.", 53),
    ("What is the cube root of 74088000? Answer with only the integer.", 420),
    ("What is 333 multiplied by 777? Answer with only the integer.", 258741),
    ("What is the square root of 15129? Answer with only the integer.", 123),
]

# ---------------------------------------------------------------------------
# EQ probes (5 questions with expected values)
# ---------------------------------------------------------------------------

EQ_PROBES = [
    (
        "Sarah just discovered her partner of 8 years has been lying about their "
        "finances. Rate her feelings of betrayal from 0-100. Answer only with a number.",
        85,
    ),
    (
        "Marcus won the championship after years of failure. Rate his joy from "
        "0-100. Answer only with a number.",
        92,
    ),
    (
        "Elena was passed over for promotion for the third time despite top reviews. "
        "Rate her frustration from 0-100. Answer only with a number.",
        82,
    ),
    (
        "James found out his childhood best friend has terminal cancer. Rate his "
        "grief from 0-100. Answer only with a number.",
        90,
    ),
    (
        "The team celebrated a surprise party for their colleague. Rate the "
        "colleague's surprise from 0-100. Answer only with a number.",
        85,
    ),
]

# ---------------------------------------------------------------------------
# Memory probes (15 factual recall questions)
# ---------------------------------------------------------------------------

MEMORY_PROBES = [
    ("What is the capital of Australia?", "canberra"),
    ("What element has the atomic number 79?", "gold"),
    ("What year did World War II end?", "1945"),
    ("Who wrote Hamlet?", "shakespeare"),
    ("What is the chemical symbol for water?", "h2o"),
    ("What planet is closest to the Sun?", "mercury"),
    ("What is the largest ocean on Earth?", "pacific"),
    ("What gas do plants absorb during photosynthesis?", "carbon dioxide"),
    ("What is the hardest natural substance on Earth?", "diamond"),
    ("How many sides does a hexagon have?", "6"),
    ("What country is the Eiffel Tower located in?", "france"),
    ("What is the speed of light in km/s? Answer as a number only.", "300000"),
    ("What organ pumps blood through the human body?", "heart"),
    ("What is the chemical formula for table salt?", "nacl"),
    ("What is the square root of 144? Answer with only the number.", "12"),
]

MEMORY_VARIANTS = {
    "carbon dioxide": ["co2", "carbon-dioxide"],
    "gold": ["au"],
    "h2o": ["water"],
    "nacl": ["sodium chloride", "salt"],
    "300000": ["299792", "3x10^5", "3 x 10^5"],
    "pacific": ["pacific ocean"],
    "canberra": [],
    "shakespeare": ["william shakespeare"],
    "mercury": [],
    "diamond": [],
    "france": ["paris"],
    "heart": [],
    "6": ["six"],
    "1945": [],
    "12": ["twelve"],
}


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def calculate_math_score(actual: int, estimate: int) -> float:
    """Partial credit math scoring from David Ng's research."""
    actual_str = str(int(actual))
    estimate_str = str(int(estimate))
    max_length = max(len(actual_str), len(estimate_str))
    actual_padded = actual_str.ljust(max_length, "0")
    estimate_padded = estimate_str.ljust(max_length, "0")
    padding_size = max_length - min(len(actual_str), len(estimate_str))
    actual_int = int(actual_padded)
    estimate_int = int(estimate_padded)
    if max(actual_int, estimate_int) == 0:
        return 0.0
    relative_diff = abs(actual_int - estimate_int) / max(actual_int, estimate_int)
    correction_factor = 1 - (padding_size / max_length)
    score = (1 - relative_diff) * correction_factor
    return max(0.0, min(score, 1.0))


def calculate_eq_score(expected: int, answer: int) -> float:
    """Score EQ response based on distance from expected value."""
    if answer < 0 or answer > 100:
        return 0.0
    distance = abs(expected - answer) / 100.0
    return max(0.0, 1.0 - distance * 2)  # 50-point tolerance for full zero


def score_memory_answer(expected: str, model_output: str) -> float:
    """Score factual recall answer via substring + variant matching.

    Returns:
        1.0: exact answer found in output
        0.5: known variant found in output
        0.0: no match
    """
    punct_table = str.maketrans("", "", string.punctuation)

    def _clean(s: str) -> str:
        return s.lower().translate(punct_table).strip()

    cleaned_output = _clean(model_output)
    cleaned_expected = _clean(expected)

    # Exact substring match
    if cleaned_expected in cleaned_output:
        return 1.0

    # Check known variants
    variants = MEMORY_VARIANTS.get(expected, [])
    for variant in variants:
        if _clean(variant) in cleaned_output:
            return 0.5

    return 0.0


def extract_number(text: str) -> int | None:
    """Extract the first integer from model output."""
    # Try to find a standalone number
    numbers = re.findall(r"-?\d+", text.strip())
    if numbers:
        return int(numbers[0])
    return None


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def run_probe(model, tokenizer, question: str, device: torch.device) -> str:
    """Run a single probe question and return the raw text output."""
    prompt = (
        f"<|im_start|>system\nYou are a precise calculator. "
        f"Answer with ONLY the number, nothing else.<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = enc["input_ids"].to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=32,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_memory_probe(model, tokenizer, question: str, device: torch.device) -> str:
    """Run a factual recall probe with a short-answer system prompt."""
    prompt = (
        f"<|im_start|>system\nAnswer with ONLY the answer, "
        f"no explanation or extra words.<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = enc["input_ids"].to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def score_config(model, tokenizer, device: torch.device, mode: str = "reasoning") -> dict:
    """Run probes based on mode and return scores.

    Modes:
        reasoning: MATH + EQ probes (original behavior)
        memory:    MEMORY probes only
        full:      all three probe sets
    """
    scores = {}

    run_reasoning = mode in ("reasoning", "full")
    run_memory = mode in ("memory", "full")

    # --- Math probes ---
    if run_reasoning:
        math_scores = []
        for question, expected in MATH_PROBES:
            text = run_probe(model, tokenizer, question, device)
            answer = extract_number(text)
            if answer is not None:
                math_scores.append(calculate_math_score(expected, answer))
            else:
                math_scores.append(0.0)
        scores["math_score"] = round(sum(math_scores) / len(math_scores), 4)
    else:
        scores["math_score"] = None

    # --- EQ probes ---
    if run_reasoning:
        eq_scores = []
        for question, expected in EQ_PROBES:
            text = run_probe(model, tokenizer, question, device)
            answer = extract_number(text)
            if answer is not None:
                eq_scores.append(calculate_eq_score(expected, answer))
            else:
                eq_scores.append(0.0)
        scores["eq_score"] = round(sum(eq_scores) / len(eq_scores), 4)
    else:
        scores["eq_score"] = None

    # --- Memory probes ---
    if run_memory:
        mem_scores = []
        for question, expected in MEMORY_PROBES:
            text = run_memory_probe(model, tokenizer, question, device)
            mem_scores.append(score_memory_answer(expected, text))
        scores["memory_score"] = round(sum(mem_scores) / len(mem_scores), 4)
    else:
        scores["memory_score"] = None

    # --- Combined score ---
    if mode == "reasoning":
        scores["combined_score"] = round(
            0.6 * scores["math_score"] + 0.4 * scores["eq_score"], 4
        )
    elif mode == "memory":
        scores["combined_score"] = scores["memory_score"]
    else:  # full
        scores["combined_score"] = round(
            0.35 * scores["math_score"]
            + 0.25 * scores["eq_score"]
            + 0.40 * scores["memory_score"],
            4,
        )

    return scores


# ---------------------------------------------------------------------------
# Cortex Loop layer manipulation
# ---------------------------------------------------------------------------

def apply_cortex_loop(model, i: int, j: int, original_layers: list) -> int:
    """Apply Cortex Loop duplication using pointer manipulation (zero VRAM)."""
    new_layers = list(original_layers[:j]) + list(original_layers[i:j]) + list(original_layers[j:])
    model.model.layers = nn.ModuleList(new_layers)
    new_count = len(new_layers)
    model.config.num_hidden_layers = new_count
    return new_count


def restore_layers(model, original_layers: list, original_count: int) -> None:
    """Restore original layer configuration."""
    model.model.layers = nn.ModuleList(list(original_layers))
    model.config.num_hidden_layers = original_count


# ---------------------------------------------------------------------------
# Heatmaps
# ---------------------------------------------------------------------------

def print_heatmap(results: list, num_layers: int, delta_key: str = "combined_delta",
                  title: str = "Cortex Loop Scan Heatmap (combined delta)") -> None:
    """Print ASCII heatmap of deltas."""
    chars = " .:-=+*#%@"

    grid = {}
    deltas = [r[delta_key] for r in results if r.get(delta_key) is not None]
    if not deltas:
        return
    min_d = min(deltas)
    max_d = max(deltas)
    rng = max_d - min_d if max_d != min_d else 1.0

    for r in results:
        if r.get(delta_key) is not None:
            grid[(r["i"], r["j"])] = r[delta_key]

    print(f"\n  {title}")
    print(f"  Min delta: {min_d:+.4f}  Max delta: {max_d:+.4f}")
    print("  X-axis: j (end layer)  Y-axis: i (start layer)\n")

    header = "  i\\j "
    for j in range(2, min(num_layers + 1, 29)):
        header += f"{j:3d}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    for i in range(num_layers - 1):
        row = f"  {i:3d} "
        for j in range(2, min(num_layers + 1, 29)):
            if (i, j) in grid:
                d = grid[(i, j)]
                idx = int((d - min_d) / rng * (len(chars) - 1))
                idx = max(0, min(idx, len(chars) - 1))
                row += f"  {chars[idx]}"
            else:
                row += "   "
        print(row)
    print()


def print_memory_heatmap(results: list, num_layers: int) -> None:
    """Print ASCII memory cortex heatmap with directional symbols."""
    grid = {}
    for r in results:
        if r.get("memory_delta") is not None:
            grid[(r["i"], r["j"])] = r["memory_delta"]

    if not grid:
        return

    deltas = list(grid.values())
    min_d = min(deltas)
    max_d = max(deltas)

    print("\n  === MEMORY CORTEX HEATMAP ===")
    print("  (+ = factual recall improves, X = degrades)")
    print(f"  Min delta: {min_d:+.4f}  Max delta: {max_d:+.4f}")
    print("  X-axis: j (end layer)  Y-axis: i (start layer)\n")

    header = "  i\\j "
    for j in range(2, min(num_layers + 1, 29)):
        header += f"{j:3d}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    for i in range(num_layers - 1):
        row = f"  {i:3d} "
        for j in range(2, min(num_layers + 1, 29)):
            if (i, j) in grid:
                d = grid[(i, j)]
                if d > 0.05:
                    ch = "+"
                elif d > 0.0:
                    ch = "."
                elif d >= -0.05:
                    ch = "~" if d == 0.0 else "-"
                else:
                    ch = "X"
                row += f"  {ch}"
            else:
                row += "   "
        print(row)
    print()


# ---------------------------------------------------------------------------
# PNG heatmap export (matplotlib)
# ---------------------------------------------------------------------------

def save_heatmap_png(results: list, num_layers: int, delta_key: str,
                     title: str, out_path: Path) -> None:
    """Save a color heatmap as PNG using matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError:
        print(f"  WARNING: matplotlib not installed. Skipping PNG export for {out_path.name}")
        return

    # Build grid
    grid = {}
    for r in results:
        val = r.get(delta_key)
        if val is not None:
            grid[(r["i"], r["j"])] = val

    if not grid:
        return

    # Determine axis ranges from data
    all_i = sorted(set(k[0] for k in grid))
    all_j = sorted(set(k[1] for k in grid))

    # Build matrix
    matrix = np.full((len(all_i), len(all_j)), np.nan)
    i_idx = {v: idx for idx, v in enumerate(all_i)}
    j_idx = {v: idx for idx, v in enumerate(all_j)}

    for (i, j), val in grid.items():
        matrix[i_idx[i], j_idx[j]] = val

    # Determine color normalization centered at 0
    deltas = [v for v in grid.values()]
    abs_max = max(abs(min(deltas)), abs(max(deltas)), 0.001)
    vmin = -abs_max
    vmax = abs_max

    fig, ax = plt.subplots(figsize=(max(14, len(all_j) * 0.35), max(10, len(all_i) * 0.25)))

    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    cmap = plt.cm.RdBu_r  # red = positive (good), blue = negative

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")

    # Axis labels: show every Nth tick to avoid crowding
    i_step = max(1, len(all_i) // 30)
    j_step = max(1, len(all_j) // 40)
    ax.set_yticks(range(0, len(all_i), i_step))
    ax.set_yticklabels([str(all_i[k]) for k in range(0, len(all_i), i_step)], fontsize=7)
    ax.set_xticks(range(0, len(all_j), j_step))
    ax.set_xticklabels([str(all_j[k]) for k in range(0, len(all_j), j_step)], fontsize=7)

    ax.set_xlabel("j (end layer)", fontsize=10)
    ax.set_ylabel("i (start layer)", fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Delta from baseline", fontsize=9)

    # Annotate best config
    best_key = max(grid, key=grid.get)
    best_val = grid[best_key]
    best_row = i_idx[best_key[0]]
    best_col = j_idx[best_key[1]]
    ax.plot(best_col, best_row, marker="*", color="gold", markersize=16,
            markeredgecolor="black", markeredgewidth=1.0)
    ax.annotate(
        f"BEST (i={best_key[0]}, j={best_key[1]})\n{best_val:+.4f}",
        xy=(best_col, best_row),
        xytext=(best_col + 2, best_row - 2),
        fontsize=8, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
    )

    # Metadata footer
    fig.text(0.5, 0.01,
             f"PRISM Cortex Loop Scan, {num_layers} layers, {len(grid)} configs, {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
             ha="center", fontsize=8, color="gray")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap saved: {out_path}")


def save_all_heatmaps(results: list, num_layers: int, mode: str, logs_dir: Path) -> None:
    """Generate and save all relevant PNG heatmaps."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # Combined heatmap (always)
    save_heatmap_png(
        results, num_layers,
        delta_key="combined_delta",
        title="Cortex Loop Scan: Combined Delta (0.35*Math + 0.25*EQ + 0.40*Memory)"
               if mode == "full" else "Cortex Loop Scan: Combined Delta",
        out_path=logs_dir / f"heatmap_combined_{timestamp}.png",
    )

    # Math delta
    if mode in ("reasoning", "full"):
        save_heatmap_png(
            results, num_layers,
            delta_key="math_delta",
            title="Cortex Loop Scan: Math Reasoning Delta",
            out_path=logs_dir / f"heatmap_math_{timestamp}.png",
        )

    # EQ delta
    if mode in ("reasoning", "full"):
        save_heatmap_png(
            results, num_layers,
            delta_key="eq_delta",
            title="Cortex Loop Scan: Emotional Quotient Delta",
            out_path=logs_dir / f"heatmap_eq_{timestamp}.png",
        )

    # Memory delta
    if mode in ("memory", "full"):
        save_heatmap_png(
            results, num_layers,
            delta_key="memory_delta",
            title="Cortex Loop Scan: Memory Cortex Delta (Factual Recall)",
            out_path=logs_dir / f"heatmap_memory_{timestamp}.png",
        )


# ---------------------------------------------------------------------------
# Post-scan summaries
# ---------------------------------------------------------------------------

def print_neuroanatomy(results: list, baseline: dict, num_layers: int, mode: str) -> None:
    """Print the anatomical summary after a scan completes."""
    model_name = getattr(config, "MODEL_ID", "Qwen2.5-7B-Instruct")

    print("\n" + "=" * 60)
    print("  === PRISM NEUROANATOMY RESULTS ===")
    print("=" * 60)
    print(f"\n  Model: {model_name} ({num_layers} layers)\n")

    # --- Reasoning cortex ---
    if mode in ("reasoning", "full"):
        reasoning_ranked = sorted(
            [r for r in results if r.get("combined_delta") is not None],
            key=lambda r: (r.get("math_delta", 0) or 0) + (r.get("eq_delta", 0) or 0),
            reverse=True,
        )
        if reasoning_ranked:
            best_r = reasoning_ranked[0]
            r_delta = (best_r.get("math_delta", 0) or 0) + (best_r.get("eq_delta", 0) or 0)
            print("  REASONING CORTEX")
            print(f"    Best block  : layers {best_r['i']} → {best_r['j']} ({best_r['block_size']} layers)")
            print(f"    Delta       : {r_delta:+.4f}")
            print("    Interpretation: Duplicating this block gives the model a")
            print("                    second pass through its core reasoning circuit.\n")

    # --- Memory cortex ---
    if mode in ("memory", "full"):
        memory_ranked = sorted(
            [r for r in results if r.get("memory_delta") is not None],
            key=lambda r: r["memory_delta"],
            reverse=True,
        )
        if memory_ranked:
            best_m = memory_ranked[0]
            print("  MEMORY CORTEX")
            print(f"    Best block  : layers {best_m['i']} → {best_m['j']} ({best_m['block_size']} layers)")
            print(f"    Delta       : {best_m['memory_delta']:+.4f}")
            print("    Interpretation: These MLP-heavy layers store factual knowledge.")
            print("                    Optimal injection point for Titans memory adapter.\n")

    # --- Seam layers ---
    if mode == "full":
        if reasoning_ranked and memory_ranked:
            best_r = reasoning_ranked[0]
            best_m = memory_ranked[0]
            print("  SEAM LAYERS (for LoRA targeting):")
            print(f"    Reasoning seam : layer {best_r['i']} → layer {best_r['j']}")
            print(f"    Memory seam    : layer {best_m['i']} → layer {best_m['j']}")
            print()


def print_titans_recommendation(results: list, mode: str) -> None:
    """Print Titans injection layer recommendation based on memory cortex."""
    if mode not in ("memory", "full"):
        return

    memory_ranked = sorted(
        [r for r in results if r.get("memory_delta") is not None],
        key=lambda r: r["memory_delta"],
        reverse=True,
    )
    if not memory_ranked:
        return

    best = memory_ranked[0]
    inject_layer = best["i"]

    print("=" * 60)
    print("  === TITANS INJECTION RECOMMENDATION ===")
    print("=" * 60)
    print(f"  Based on memory cortex scan:")
    print(f"  Inject Titans adapter at layer {inject_layer} (entry to memory cortex)")
    print(f"  This replaces the current embedding-layer injection point.")
    print(f"  Update TITANS_MEMORY_LAYERS={inject_layer} in your .env to apply this.")
    print()


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cortex Loop Scanner for Qwen2.5-7B")
    parser.add_argument("--fast", action="store_true", help="Only test block sizes 2-8")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted scan")
    parser.add_argument("--model-path", type=str, default=None, help="Override model path")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["reasoning", "memory", "full"],
        default="reasoning",
        help="Scan mode: reasoning (math+EQ), memory (factual recall), full (all three)",
    )
    args = parser.parse_args()

    mode = args.mode
    model_path = args.model_path or config.MODEL_LOCAL_PATH
    results_path = Path("logs/cortex_loop_scan_results.json")
    config_out_path = Path("config/rys_config.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    config_out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Scan mode: {mode}")
    probe_counts = {
        "reasoning": f"{len(MATH_PROBES)} math + {len(EQ_PROBES)} EQ",
        "memory": f"{len(MEMORY_PROBES)} memory",
        "full": f"{len(MATH_PROBES)} math + {len(EQ_PROBES)} EQ + {len(MEMORY_PROBES)} memory",
    }
    print(f"Probes: {probe_counts[mode]}\n")

    # Load existing results for resume
    existing_results = {}
    if args.resume and results_path.exists():
        data = json.loads(results_path.read_text())
        saved_mode = data.get("mode", "reasoning")
        if saved_mode != mode:
            print(f"WARNING: Saved results used mode '{saved_mode}', current mode is '{mode}'.")
            print("         Ignoring saved results (modes must match to resume).\n")
        else:
            for r in data.get("results", []):
                existing_results[(r["i"], r["j"])] = r
            if "baseline" in data:
                existing_results["baseline"] = data["baseline"]
            print(f"Resuming: {len(existing_results)} configs already completed.\n")

    # Load model
    print(f"Loading model from {model_path}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    device = next(model.parameters()).device
    num_layers = model.config.num_hidden_layers
    print(f"Model loaded: {num_layers} layers on {device}\n")

    # Save original state
    original_layers = list(model.model.layers)
    original_count = num_layers

    # Generate config pairs
    pairs = []
    for i in range(num_layers):
        for j in range(i + 2, num_layers + 1):
            block_size = j - i
            if args.fast and (block_size < 2 or block_size > 8):
                continue
            pairs.append((i, j))

    total = len(pairs)
    already_done = sum(1 for p in pairs if p in existing_results)
    remaining = total - already_done

    # Estimate time per config based on mode
    probes_per_config = {"reasoning": 15, "memory": 15, "full": 30}
    est_per_config = probes_per_config[mode] * 1.7  # ~1.7s per probe
    est_seconds = remaining * est_per_config
    est_minutes = est_seconds / 60

    print(f"Configs to test: {total} total, {already_done} done, {remaining} remaining")
    print(f"Estimated runtime: ~{est_minutes:.0f} minutes ({est_seconds / 3600:.1f} hours)\n")

    # Run baseline first
    if "baseline" not in existing_results:
        print("Running baseline (unmodified model)...")
        baseline = score_config(model, tokenizer, device, mode=mode)
        parts = []
        if baseline["math_score"] is not None:
            parts.append(f"math={baseline['math_score']:.4f}")
        if baseline["eq_score"] is not None:
            parts.append(f"eq={baseline['eq_score']:.4f}")
        if baseline["memory_score"] is not None:
            parts.append(f"memory={baseline['memory_score']:.4f}")
        parts.append(f"combined={baseline['combined_score']:.4f}")
        print(f"  Baseline: {' '.join(parts)}\n")
    else:
        baseline = existing_results["baseline"]
        print(f"  Baseline (cached): combined={baseline['combined_score']:.4f}\n")

    # Sweep
    results = []
    for idx, (i, j) in enumerate(pairs):
        if (i, j) in existing_results:
            results.append(existing_results[(i, j)])
            continue

        # Apply Cortex Loop config
        new_count = apply_cortex_loop(model, i, j, original_layers)

        # Score
        try:
            scores = score_config(model, tokenizer, device, mode=mode)
        except Exception as e:
            print(f"  ERROR (i={i}, j={j}): {e}")
            scores = {
                "math_score": 0.0 if mode in ("reasoning", "full") else None,
                "eq_score": 0.0 if mode in ("reasoning", "full") else None,
                "memory_score": 0.0 if mode in ("memory", "full") else None,
                "combined_score": 0.0,
            }

        # Restore
        restore_layers(model, original_layers, original_count)
        torch.cuda.empty_cache()

        # Compute deltas
        def _delta(key):
            if scores.get(key) is None or baseline.get(key) is None:
                return None
            return round(scores[key] - baseline[key], 4)

        math_delta = _delta("math_score")
        eq_delta = _delta("eq_score")
        memory_delta = _delta("memory_score")
        combined_delta = round(scores["combined_score"] - baseline["combined_score"], 4)

        result = {
            "i": i,
            "j": j,
            "block_size": j - i,
            "effective_layers": original_count + (j - i),
            **scores,
            "math_delta": math_delta,
            "eq_delta": eq_delta,
            "memory_delta": memory_delta,
            "combined_delta": combined_delta,
        }
        results.append(result)

        done = idx + 1
        parts = []
        if scores["math_score"] is not None:
            parts.append(f"math={scores['math_score']:.4f}")
        if scores["eq_score"] is not None:
            parts.append(f"eq={scores['eq_score']:.4f}")
        if scores["memory_score"] is not None:
            parts.append(f"mem={scores['memory_score']:.4f}")

        print(
            f"  [{done}/{total}] (i={i:2d}, j={j:2d}) bs={j-i} | "
            f"{' '.join(parts)} | delta={combined_delta:+.4f}"
        )

        # Save progress periodically
        if done % 10 == 0:
            _save_results(results_path, results, baseline, mode)

    # Final save
    _save_results(results_path, results, baseline, mode)

    # Sort by combined delta
    ranked = sorted(results, key=lambda r: r["combined_delta"], reverse=True)

    print("\n" + "=" * 60)
    print("  TOP 10 CONFIGURATIONS")
    print("=" * 60)
    for k, r in enumerate(ranked[:10]):
        parts = []
        if r["math_delta"] is not None:
            parts.append(f"math={r['math_delta']:+.4f}")
        if r["eq_delta"] is not None:
            parts.append(f"eq={r['eq_delta']:+.4f}")
        if r["memory_delta"] is not None:
            parts.append(f"mem={r['memory_delta']:+.4f}")
        print(
            f"  #{k+1}: (i={r['i']:2d}, j={r['j']:2d}) bs={r['block_size']} | "
            f"{' '.join(parts)} | combined={r['combined_delta']:+.4f}"
        )

    best = ranked[0]
    print(f"\n  BEST CONFIG: (i={best['i']}, j={best['j']}), "
          f"block_size={best['block_size']}, delta={best['combined_delta']:+.4f}")

    # --- Save best config (structured by cortex) ---
    now_iso = datetime.utcnow().isoformat()

    cl_config_out: dict = {}

    # Reasoning best (by math + eq delta)
    if mode in ("reasoning", "full"):
        reasoning_ranked = sorted(
            results,
            key=lambda r: (r.get("math_delta") or 0) + (r.get("eq_delta") or 0),
            reverse=True,
        )
        best_r = reasoning_ranked[0]
        cl_config_out["reasoning"] = {
            "i": best_r["i"],
            "j": best_r["j"],
            "block_size": best_r["block_size"],
            "math_delta": best_r["math_delta"],
            "eq_delta": best_r["eq_delta"],
            "combined_delta": best_r["combined_delta"],
        }

    # Memory best (by memory_delta)
    if mode in ("memory", "full"):
        memory_ranked = sorted(
            [r for r in results if r.get("memory_delta") is not None],
            key=lambda r: r["memory_delta"],
            reverse=True,
        )
        best_m = memory_ranked[0]
        cl_config_out["memory"] = {
            "i": best_m["i"],
            "j": best_m["j"],
            "block_size": best_m["block_size"],
            "memory_delta": best_m["memory_delta"],
            "combined_delta": best_m["combined_delta"],
        }

    cl_config_out["scan_mode"] = mode
    cl_config_out["scan_date"] = now_iso
    cl_config_out["baseline"] = baseline

    config_out_path.write_text(json.dumps(cl_config_out, indent=2))
    print(f"\n  Config saved to: {config_out_path}")
    print(f"  Full results at: {results_path}")

    # --- Heatmaps ---
    print_heatmap(results, num_layers, "combined_delta", "Cortex Loop Scan Heatmap (combined delta)")

    if mode in ("memory", "full"):
        print_memory_heatmap(results, num_layers)

    # --- PNG heatmaps ---
    save_all_heatmaps(results, num_layers, mode, results_path.parent)

    # --- Neuroanatomy summary ---
    print_neuroanatomy(results, baseline, num_layers, mode)

    # --- Titans injection recommendation ---
    print_titans_recommendation(results, mode)

    # --- Next steps ---
    print("Next steps:")
    print("  1. python scripts/cortex_loop_apply.py")
    print("  2. Set CORTEX_LOOP_ENABLED=true and CORTEX_LOOP_SCAN_COMPLETE=true in .env")
    step = 3
    if mode in ("memory", "full"):
        print(f"  {step}. Set TITANS_MEMORY_LAYERS=<recommended layer> in .env")
        step += 1
    print(f"  {step}. Restart PRISM server")


def _save_results(path: Path, results: list, baseline: dict, mode: str) -> None:
    """Save scan results to JSON (for resume support)."""
    data = {
        "mode": mode,
        "baseline": baseline,
        "results": results,
        "saved_at": datetime.utcnow().isoformat(),
    }
    path.write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
