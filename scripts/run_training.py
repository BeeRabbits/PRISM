"""
Manual training trigger script.

Usage:
    python scripts/run_training.py
    python scripts/run_training.py --min-fitness 0.7
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import typer
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.experience_log import get_experience_logger
from model.loader import ModelLoader
from training.lora_trainer import LoRAContinualTrainer

console = Console()
cli = typer.Typer(add_completion=False)


@cli.command()
def main(
    min_fitness: float = typer.Option(
        config.TRAINING_MIN_FITNESS, help="Minimum fitness score for training data."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Check episode count without training."
    ),
) -> None:
    """Manually trigger a LoRA fine-tuning run."""

    async def _check() -> int:
        exp_log = get_experience_logger()
        await exp_log.init()
        return await exp_log.count_high_fitness(min_score=min_fitness)

    n = asyncio.run(_check())
    console.print(f"High-fitness episodes available: [bold]{n}[/bold]")
    console.print(f"Minimum required             : [bold]{config.TRAINING_MIN_EPISODES}[/bold]")

    if n < config.TRAINING_MIN_EPISODES:
        console.print("[yellow]Not enough data. Aborting.[/yellow]")
        raise typer.Exit(1)

    if dry_run:
        console.print("[yellow]--dry-run set. Skipping actual training.[/yellow]")
        return

    console.print("[bold green]Loading model...[/bold green]")
    loader = ModelLoader()
    loader.load()

    console.print("[bold green]Starting training...[/bold green]")
    trainer = LoRAContinualTrainer(
        model=loader.model,
        tokenizer=loader.tokenizer,
    )
    result = trainer.train()

    console.print()
    console.print("[bold]Training result:[/bold]")
    console.print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    cli()
