"""Load the MIRROR context file with a sensible fallback."""

from pathlib import Path

import config


def load_mirror_context() -> str:
    """
    Read the user's MIRROR context file (config.MIRROR_CONTEXT_FILE).

    Returns the file contents, or a generic fallback string if the
    file does not exist.
    """
    path = Path(config.MIRROR_CONTEXT_FILE)
    if path.exists():
        return path.read_text().strip()
    return (
        "The user is building a personalized AI assistant. "
        "Evaluate responses for helpfulness, accuracy, and personality fit."
    )
