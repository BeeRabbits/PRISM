"""
Simple before/after benchmark for PRISM.

Runs a fixed set of prompts against the server and records responses.
Run BEFORE a training cycle to capture baseline, then AFTER to compare.

Usage:
    python scripts/benchmark.py --tag before
    python scripts/benchmark.py --tag after
    python scripts/benchmark.py --compare before after
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from prism_inside_client import PrismInsideClient

console = Console()
cli = typer.Typer(add_completion=False)

_BENCHMARK_PROMPTS = [
    "What is 2 + 2?",
    "Explain the concept of entropy in one sentence.",
    "Write a haiku about memory.",
    "What is the capital of France?",
    "Describe yourself in 3 words.",
]

_RESULTS_DIR = Path("data/benchmarks")


@cli.command("run")
def run_benchmark(
    tag: str = typer.Option(..., help="Label for this benchmark run (e.g. 'before', 'after')."),
    base_url: str = typer.Option("http://localhost:8000", help="Server base URL."),
) -> None:
    """Run the benchmark prompts and save results."""
    client = PrismInsideClient(base_url, session_id=f"bench_{tag}")
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for i, prompt in enumerate(_BENCHMARK_PROMPTS, 1):
        console.print(f"[{i}/{len(_BENCHMARK_PROMPTS)}] {prompt[:60]}...")
        try:
            r = client.chat(prompt)
            results.append({
                "prompt": prompt,
                "response": r["response"],
                "episode_id": r["episode_id"],
            })
            console.print(f"  → {r['response'][:80]}...")
        except Exception as e:
            results.append({"prompt": prompt, "error": str(e)})
            console.print(f"  [red]Error: {e}[/red]")

    out_file = _RESULTS_DIR / f"{tag}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with out_file.open("w") as f:
        json.dump({"tag": tag, "timestamp": datetime.utcnow().isoformat(), "results": results}, f, indent=2)
    console.print(f"\n[green]Results saved to {out_file}[/green]")


@cli.command("compare")
def compare_benchmarks(
    before_tag: str = typer.Argument(..., help="Tag of the 'before' benchmark."),
    after_tag: str = typer.Argument(..., help="Tag of the 'after' benchmark."),
) -> None:
    """Compare two benchmark results side by side."""
    before_files = sorted(_RESULTS_DIR.glob(f"{before_tag}_*.json"))
    after_files = sorted(_RESULTS_DIR.glob(f"{after_tag}_*.json"))

    if not before_files or not after_files:
        console.print("[red]Could not find benchmark files for the given tags.[/red]")
        raise typer.Exit(1)

    before = json.loads(before_files[-1].read_text())
    after = json.loads(after_files[-1].read_text())

    table = Table(title=f"Benchmark: {before_tag} vs {after_tag}", show_lines=True)
    table.add_column("Prompt", style="bold", width=30)
    table.add_column(f"Before ({before_tag})", width=40)
    table.add_column(f"After ({after_tag})", width=40)

    before_map = {r["prompt"]: r.get("response", r.get("error", "")) for r in before["results"]}
    after_map = {r["prompt"]: r.get("response", r.get("error", "")) for r in after["results"]}

    for prompt in _BENCHMARK_PROMPTS:
        b = before_map.get(prompt, "-")[:80]
        a = after_map.get(prompt, "-")[:80]
        table.add_row(prompt[:30], b, a)

    console.print(table)


if __name__ == "__main__":
    cli()
