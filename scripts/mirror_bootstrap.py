"""
MIRROR Bootstrap: import prior conversation history as pre-scored episodes.

Usage:
    python scripts/mirror_bootstrap.py path/to/conversation.md

The file should contain conversation turns in this format:

    User: message here
    Assistant: response here

    User: another message
    Assistant: another response

Or with markdown headers:

    ## User
    message here

    ## Assistant
    response here

Episodes are created with oracle_score=4 (good quality from a strong source),
mirror_iteration=0, auto_scored=True.
"""

from __future__ import annotations

import asyncio
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: E402
from data.experience_log import ExperienceLogger, EpisodeRow  # noqa: E402


def parse_conversations(text: str) -> list[tuple[str, str]]:
    """
    Parse a text file into (user_message, assistant_response) pairs.

    Supports multiple formats:
      - "User: ... \\nAssistant: ..."
      - "## User\\n...\\n## Assistant\\n..."
      - "Human: ... \\nAssistant: ..."
    """
    pairs = []

    # Try "User:/Human:" + "Assistant:" pattern
    pattern = re.compile(
        r"(?:^|\n)\s*(?:User|Human)\s*:\s*(.*?)(?=\n\s*(?:Assistant|AI)\s*:)",
        re.DOTALL | re.IGNORECASE,
    )
    assistant_pattern = re.compile(
        r"(?:^|\n)\s*(?:Assistant|AI)\s*:\s*(.*?)(?=\n\s*(?:User|Human)\s*:|$)",
        re.DOTALL | re.IGNORECASE,
    )

    user_matches = list(pattern.finditer(text))
    assistant_matches = list(assistant_pattern.finditer(text))

    if user_matches and assistant_matches:
        for u, a in zip(user_matches, assistant_matches):
            user_msg = u.group(1).strip()
            asst_msg = a.group(1).strip()
            if user_msg and asst_msg:
                pairs.append((user_msg, asst_msg))
        return pairs

    # Fallback: markdown headers
    sections = re.split(r"\n##\s+", text)
    user_msg = None
    for section in sections:
        header_match = re.match(r"(User|Human|Assistant|AI)\s*\n(.*)", section, re.DOTALL | re.IGNORECASE)
        if header_match:
            role = header_match.group(1).lower()
            content = header_match.group(2).strip()
            if role in ("user", "human"):
                user_msg = content
            elif role in ("assistant", "ai") and user_msg:
                pairs.append((user_msg, content))
                user_msg = None

    return pairs


async def bootstrap(file_path: str) -> None:
    """Import conversation file as MIRROR bootstrap episodes."""
    path = Path(file_path)
    if not path.exists():
        print(f"Error: file not found: {path}")
        sys.exit(1)

    text = path.read_text()
    pairs = parse_conversations(text)

    if not pairs:
        print("Error: could not parse any conversation pairs from the file.")
        print("Expected format: 'User: message\\nAssistant: response'")
        sys.exit(1)

    print(f"Parsed {len(pairs)} conversation pairs.")

    logger = ExperienceLogger()
    await logger.init()

    count = 0
    for user_msg, asst_msg in pairs:
        episode_id = str(uuid.uuid4())
        row = EpisodeRow(
            id=episode_id,
            session_id="mirror_bootstrap",
            user_message=user_msg,
            assistant_response=asst_msg,
            full_prompt="[bootstrap import]",
            memory_context_used="",
            timestamp=datetime.utcnow(),
            fitness_score=0.75,  # (4-1)/4 = 0.75 on the 0-1 scale
            oracle_score=4.0,
            self_score=None,
            mirror_delta=None,
            mirror_iteration=0,
            auto_scored=True,
            oracle_reasoning="Bootstrap import from high-quality conversation source.",
        )
        async with logger._session_factory() as session:
            session.add(row)
            await session.commit()
        count += 1

    print(f"\nBootstrap complete:")
    print(f"  Episodes imported : {count}")
    print(f"  Oracle score      : 4.0 (pre-set)")
    print(f"  Fitness score     : 0.75")
    print(f"  Session ID        : mirror_bootstrap")
    print(f"  Mirror iteration  : 0")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/mirror_bootstrap.py <conversation_file>")
        print("\nThe file should contain 'User: ... / Assistant: ...' pairs.")
        sys.exit(1)

    asyncio.run(bootstrap(sys.argv[1]))
