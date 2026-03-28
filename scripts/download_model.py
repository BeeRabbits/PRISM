"""
Download Qwen2.5-7B-Instruct from HuggingFace and verify the download.

Usage:
    python scripts/download_model.py
"""

import hashlib
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from huggingface_hub import snapshot_download
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def _verify_download(local_path: Path) -> bool:
    """Check that critical model files are present."""
    required = [
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    ]
    missing = [f for f in required if not (local_path / f).exists()]
    if missing:
        console.print(f"[red]Missing files: {missing}[/red]")
        return False

    # At least one safetensors shard must exist
    shards = list(local_path.glob("*.safetensors"))
    if not shards:
        console.print("[red]No .safetensors files found. Download may be incomplete.[/red]")
        return False

    total_bytes = sum(s.stat().st_size for s in shards)
    console.print(
        f"[green]Found {len(shards)} shard(s), "
        f"total size: {total_bytes / 1e9:.2f} GB[/green]"
    )
    return True


def download_model() -> None:
    """Download the model from HuggingFace Hub."""
    local_path = Path(config.MODEL_LOCAL_PATH)

    if local_path.exists() and _verify_download(local_path):
        console.print(
            f"[yellow]Model already downloaded at {local_path}. Skipping.[/yellow]"
        )
        return

    local_path.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Downloading {config.MODEL_ID}...[/bold]")
    console.print(f"  Destination : {local_path.resolve()}")
    console.print(f"  HF Token    : {'set' if config.HF_TOKEN else 'NOT SET. May fail for gated models'}")
    console.print()

    if not config.HF_TOKEN:
        console.print(
            "[yellow]Warning: HF_TOKEN not set. "
            "Set it in .env if the model is gated.[/yellow]"
        )

    try:
        snapshot_download(
            repo_id=config.MODEL_ID,
            local_dir=str(local_path),
            token=config.HF_TOKEN or None,
            ignore_patterns=["*.pt", "original/*"],  # skip redundant formats
        )
    except Exception as e:
        console.print(f"[red]Download failed: {e}[/red]")
        sys.exit(1)

    console.print()
    if _verify_download(local_path):
        console.print("[bold green]Download verified successfully.[/bold green]")
    else:
        console.print("[red]Download verification failed. Try re-running.[/red]")
        sys.exit(1)


if __name__ == "__main__":
    download_model()
