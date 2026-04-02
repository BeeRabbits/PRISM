"""
PRISM Metrics Dashboard: visualize training progress and MIRROR convergence.

Reads nightly run logs from logs/ and the episode database to generate
charts showing loss curves, delta convergence, dataset composition,
and training history over time.

Usage:
    python scripts/metrics_dashboard.py                # generate all charts
    python scripts/metrics_dashboard.py --output dir   # save to custom directory
    python scripts/metrics_dashboard.py --show         # display instead of save
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for server use
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except ImportError:
    print("Error: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)


# Consistent style
COLORS = {
    "lora_loss": "#2196F3",
    "eval_loss": "#FF5722",
    "titans_loss": "#4CAF50",
    "delta": "#9C27B0",
    "episodes": "#FF9800",
    "convergence_line": "#F44336",
    "personal": "#2196F3",
    "general": "#4CAF50",
    "replay": "#FF9800",
    "semantic": "#9C27B0",
}


def load_nightly_logs(log_dir: Path) -> list[dict]:
    """Load all nightly run JSON logs sorted by date."""
    logs = []
    for path in sorted(log_dir.glob("nightly_run_*.json")):
        try:
            data = json.loads(path.read_text())
            data["_file"] = path.name
            logs.append(data)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Warning: could not read {path.name}: {e}")
    return logs


def parse_timestamp(ts: str) -> datetime:
    """Parse ISO timestamp from logs."""
    for fmt in ["%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"]:
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return datetime.now()


def plot_loss_curves(logs: list[dict], output_dir: Path, show: bool = False) -> None:
    """Plot LoRA train/eval loss and Titans loss over training runs."""
    run_labels = []
    train_losses = []
    eval_losses = []
    titans_losses = []

    run_num = 0
    for log in logs:
        lora = log.get("lora") or {}
        titans = log.get("titans") or {}
        if lora.get("status") != "completed":
            continue

        # Skip runs with NaN or missing eval loss
        eval_loss = lora.get("eval_loss")
        train_loss = lora.get("train_loss")
        if eval_loss is None or train_loss is None:
            continue
        # Handle NaN (JSON NaN gets parsed as float nan)
        if math.isnan(eval_loss) or math.isnan(train_loss):
            continue

        run_num += 1
        ts = parse_timestamp(log.get("started_at", ""))
        run_labels.append(f"Run {run_num}\n{ts.strftime('%m/%d')}")
        train_losses.append(train_loss)
        eval_losses.append(eval_loss)
        titans_losses.append(titans.get("avg_loss") if titans.get("status") == "completed" else None)

    if not run_labels:
        print("  No completed training runs found. Skipping loss curves.")
        return

    x = list(range(len(run_labels)))
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(x, train_losses, "o-", color=COLORS["lora_loss"], label="LoRA Train Loss", linewidth=2, markersize=8)
    ax1.plot(x, eval_losses, "s-", color=COLORS["eval_loss"], label="LoRA Eval Loss", linewidth=2, markersize=8)

    titans_valid = [(i, t) for i, t in zip(x, titans_losses) if t is not None]
    if titans_valid:
        ti, tl = zip(*titans_valid)
        ax1.plot(ti, tl, "^-", color=COLORS["titans_loss"], label="Titans Avg Loss", linewidth=2, markersize=8)

    # Annotate each eval loss point
    for i, val in enumerate(eval_losses):
        ax1.annotate(f"{val:.4f}", (x[i], val), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9, color=COLORS["eval_loss"])

    ax1.set_xticks(x)
    ax1.set_xticklabels(run_labels, fontsize=10)
    ax1.set_xlabel("Training Run", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("PRISM Training Loss Over Time", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    if show:
        plt.show()
    else:
        path = output_dir / "loss_curves.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close()


def plot_delta_convergence(logs: list[dict], output_dir: Path, show: bool = False) -> None:
    """Plot MIRROR delta convergence over time."""
    run_labels = []
    deltas = []
    episode_counts = []

    run_num = 0
    for log in logs:
        mirror = log.get("mirror") or {}
        lora = log.get("lora") or {}
        avg_delta = mirror.get("rolling_avg_delta") or lora.get("mirror_avg_delta")
        if avg_delta is None:
            continue

        run_num += 1
        ts = parse_timestamp(log.get("started_at", ""))
        run_labels.append(f"Run {run_num}\n{ts.strftime('%m/%d')}")
        deltas.append(avg_delta)
        episode_counts.append(mirror.get("auto_scored_count") or lora.get("episodes_used", 0))

    if not run_labels:
        print("  No MIRROR delta data found. Skipping convergence plot.")
        return

    x = list(range(len(run_labels)))
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Delta line
    ax1.plot(x, deltas, "o-", color=COLORS["delta"], linewidth=2.5, markersize=8, label="Rolling Avg Delta")

    # Convergence threshold
    ax1.axhline(y=0.30, color=COLORS["convergence_line"], linestyle="--", linewidth=1.5, alpha=0.7, label="Convergence Threshold (0.30)")

    # Fill between current delta and threshold
    ax1.fill_between(x, deltas, 0.30, alpha=0.1, color=COLORS["delta"])

    # Annotate each delta value
    for i, val in enumerate(deltas):
        ax1.annotate(f"{val:.3f}", (x[i], val), textcoords="offset points",
                     xytext=(0, 12), ha="center", fontsize=10, fontweight="bold",
                     color=COLORS["delta"])

    ax1.set_xticks(x)
    ax1.set_xticklabels(run_labels, fontsize=10)
    ax1.set_xlabel("Training Run", fontsize=12)
    ax1.set_ylabel("MIRROR Delta", fontsize=12)
    ax1.set_title("MIRROR Delta Convergence", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(deltas) * 1.3)

    # Add episode count as secondary axis
    ax2 = ax1.twinx()
    ax2.bar(x, episode_counts, alpha=0.15, color=COLORS["episodes"], width=0.5, label="Episode Count")
    ax2.set_ylabel("Total Episodes", fontsize=12, color=COLORS["episodes"])
    ax2.tick_params(axis="y", labelcolor=COLORS["episodes"])

    plt.tight_layout()
    if show:
        plt.show()
    else:
        path = output_dir / "delta_convergence.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close()


def plot_episodes_over_time(logs: list[dict], output_dir: Path, show: bool = False) -> None:
    """Plot episode count growth over training runs."""
    run_labels = []
    counts = []

    run_num = 0
    for log in logs:
        lora = log.get("lora") or {}
        mirror = log.get("mirror") or {}
        count = mirror.get("auto_scored_count") or lora.get("episodes_used")
        if count is None:
            continue

        run_num += 1
        ts = parse_timestamp(log.get("started_at", ""))
        run_labels.append(f"Run {run_num}\n{ts.strftime('%m/%d')}")
        counts.append(count)

    if not run_labels:
        print("  No episode data found. Skipping episode growth plot.")
        return

    x = list(range(len(run_labels)))
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.fill_between(x, counts, alpha=0.3, color=COLORS["episodes"])
    ax.plot(x, counts, "o-", color=COLORS["episodes"], linewidth=2.5, markersize=8)

    # Annotate each point
    for i, val in enumerate(counts):
        ax.annotate(f"{val:,}", (x[i], val), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=10, fontweight="bold",
                    color=COLORS["episodes"])

    ax.set_xticks(x)
    ax.set_xticklabels(run_labels, fontsize=10)
    ax.set_xlabel("Training Run", fontsize=12)
    ax.set_ylabel("Total Episodes", fontsize=12)
    ax.set_title("PRISM Episode Growth", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if show:
        plt.show()
    else:
        path = output_dir / "episode_growth.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close()


def plot_training_summary(logs: list[dict], output_dir: Path, show: bool = False) -> None:
    """Generate a summary card with key metrics from the latest run."""
    completed = [l for l in logs if (l.get("lora") or {}).get("status") == "completed"]
    if not completed:
        print("  No completed runs. Skipping summary.")
        return

    # Use the most recent run with a valid eval_loss
    valid = [l for l in completed if l.get("lora", {}).get("eval_loss") is not None]
    latest = valid[-1] if valid else completed[-1]
    lora = latest.get("lora", {})
    titans = latest.get("titans", {})
    mirror = latest.get("mirror", {})
    consolidation = latest.get("consolidation", {})
    recall = latest.get("active_recall", {})

    # Compute improvement if we have at least 2 runs
    eval_improvement = None
    if len(completed) >= 2:
        prev_eval = completed[-2].get("lora", {}).get("eval_loss")
        curr_eval = lora.get("eval_loss")
        if prev_eval and curr_eval:
            eval_improvement = prev_eval - curr_eval

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis("off")

    title = "PRISM Training Dashboard"
    ax.text(0.5, 0.95, title, transform=ax.transAxes, fontsize=18,
            fontweight="bold", ha="center", va="top")

    run_date = parse_timestamp(latest.get("started_at", "")).strftime("%Y-%m-%d %H:%M")
    ax.text(0.5, 0.90, f"Latest run: {run_date}  |  Total runs: {len(completed)}",
            transform=ax.transAxes, fontsize=11, ha="center", va="top", color="gray")

    # Metrics in a grid layout
    metrics = []
    train_loss = lora.get("train_loss")
    eval_loss = lora.get("eval_loss")
    metrics.append(("LoRA Train Loss", f"{train_loss:.4f}" if train_loss is not None else "N/A"))
    metrics.append(("LoRA Eval Loss", f"{eval_loss:.4f}" if eval_loss is not None else "N/A"))
    if eval_improvement:
        direction = "better" if eval_improvement > 0 else "worse"
        metrics.append(("Eval Change", f"{abs(eval_improvement):.4f} {direction}"))
    metrics.append(("Episodes Used", f"{lora.get('episodes_used', 0):,}"))
    if titans.get("status") == "completed":
        metrics.append(("Titans Avg Loss", f"{titans.get('avg_loss', 0):.4f}"))
    if mirror.get("rolling_avg_delta") is not None:
        metrics.append(("MIRROR Delta", f"{mirror['rolling_avg_delta']:.3f}"))
        metrics.append(("Delta Sample", f"{mirror.get('delta_sample_size', 0)} episodes"))
    if consolidation:
        metrics.append(("Semantics Created", str(consolidation.get("semantics_created", 0))))
    if recall:
        metrics.append(("Recall Weak", f"{recall.get('weak_count', 0)}/{recall.get('total_tested', 0)}"))

    y_start = 0.78
    for i, (label, value) in enumerate(metrics):
        row = i // 2
        col = i % 2
        x = 0.15 if col == 0 else 0.55
        y = y_start - row * 0.08

        ax.text(x, y, label, transform=ax.transAxes, fontsize=12, color="gray", va="top")
        ax.text(x + 0.25, y, value, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")

    # Convergence progress bar
    delta = mirror.get("rolling_avg_delta")
    if delta is not None:
        bar_y = 0.12
        ax.text(0.5, bar_y + 0.06, "Convergence Progress", transform=ax.transAxes,
                fontsize=12, ha="center", fontweight="bold")

        # Progress: delta started around 2.5, target is 0.30
        max_delta = 2.5
        progress = max(0, min(1, (max_delta - delta) / (max_delta - 0.30)))

        bar_left = 0.1
        bar_width = 0.8
        bar_height = 0.03

        # Background bar
        bg = plt.Rectangle((bar_left, bar_y), bar_width, bar_height,
                           transform=ax.transAxes, facecolor="#E0E0E0", edgecolor="none")
        ax.add_patch(bg)

        # Progress bar
        bar_color = "#4CAF50" if progress > 0.7 else "#FF9800" if progress > 0.3 else "#F44336"
        fg = plt.Rectangle((bar_left, bar_y), bar_width * progress, bar_height,
                           transform=ax.transAxes, facecolor=bar_color, edgecolor="none")
        ax.add_patch(fg)

        ax.text(0.5, bar_y - 0.02, f"{progress * 100:.1f}% to convergence (delta {delta:.3f} / target 0.30)",
                transform=ax.transAxes, fontsize=10, ha="center", color="gray")

    plt.tight_layout()
    if show:
        plt.show()
    else:
        path = output_dir / "training_summary.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close()


def plot_consolidation_history(logs: list[dict], output_dir: Path, show: bool = False) -> None:
    """Plot dream consolidation semantics created over time."""
    run_labels = []
    created = []
    promoted = []
    clusters_found = []

    run_num = 0
    for log in logs:
        cons = log.get("consolidation")
        if not cons or cons.get("clusters_found") is None:
            continue

        run_num += 1
        ts = parse_timestamp(log.get("started_at", ""))
        run_labels.append(f"Run {run_num}\n{ts.strftime('%m/%d')}")
        created.append(cons.get("semantics_created", 0))
        promoted.append(cons.get("semantics_promoted", 0))
        clusters_found.append(cons.get("clusters_found", 0))

    if not run_labels:
        print("  No consolidation data found. Skipping.")
        return

    x = list(range(len(run_labels)))
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(x, clusters_found, width=0.5, color=COLORS["delta"], alpha=0.3, label="Clusters Found")
    ax.bar(x, created, width=0.5, color=COLORS["semantic"], alpha=0.8, label="Semantics Created")
    if any(p > 0 for p in promoted):
        ax.bar(x, promoted, width=0.5, color=COLORS["replay"], alpha=0.8,
               bottom=created, label="Semantics Promoted")

    # Annotate clusters found
    for i, val in enumerate(clusters_found):
        ax.annotate(f"{val} clusters", (x[i], val), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=9, color=COLORS["delta"])

    ax.set_xticks(x)
    ax.set_xticklabels(run_labels, fontsize=10)
    ax.set_xlabel("Training Run", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Dream Consolidation: Clusters and Semantic Memories", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if show:
        plt.show()
    else:
        path = output_dir / "consolidation_history.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="PRISM Metrics Dashboard")
    parser.add_argument("--output", type=str, default="logs/dashboard", help="Output directory for charts")
    parser.add_argument("--logs", type=str, default="logs", help="Directory containing nightly run logs")
    parser.add_argument("--show", action="store_true", help="Display charts instead of saving")
    args = parser.parse_args()

    log_dir = Path(args.logs)
    output_dir = Path(args.output)

    if not log_dir.exists():
        print(f"Error: log directory {log_dir} not found.")
        print("Run at least one training cycle first.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n  PRISM Metrics Dashboard")
    print("  " + "=" * 40)

    logs = load_nightly_logs(log_dir)
    print(f"  Loaded {len(logs)} nightly run logs.\n")

    if not logs:
        print("  No logs found. Run training first.")
        sys.exit(0)

    print("  Generating charts...")
    plot_loss_curves(logs, output_dir, args.show)
    plot_delta_convergence(logs, output_dir, args.show)
    plot_episodes_over_time(logs, output_dir, args.show)
    plot_training_summary(logs, output_dir, args.show)
    plot_consolidation_history(logs, output_dir, args.show)

    if not args.show:
        print(f"\n  All charts saved to {output_dir}/")
        print("  Files:")
        for f in sorted(output_dir.glob("*.png")):
            print(f"    - {f.name}")

    print("\n  " + "=" * 40)
    print("  Dashboard complete.\n")


if __name__ == "__main__":
    main()
