"""
MIRROR Autopilot: Claude simulates conversations with PRISM as the user.

Claude generates messages based on mirror_context.md (your personal context),
PRISM responds, and MIRROR auto-scores each exchange. Watch them
chat in real-time while training data accumulates.

Usage:
    python scripts/mirror_autopilot.py                    # 20 turns, mixed topics
    python scripts/mirror_autopilot.py --turns 50         # 50 turns
    python scripts/mirror_autopilot.py --topic fitness    # focused topic
    python scripts/mirror_autopilot.py --watch            # slower output, easier to read
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import random
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
import config  # noqa: E402
from utils.mirror_context import load_mirror_context  # noqa: E402

# Topics Claude will cycle through (personalized via mirror_context.md at runtime)
TOPICS = [
    "family life, parenting, household logistics",
    "relationships, future planning, living situation",
    "career growth, current role, professional ambitions",
    "fitness, working out, health goals, diet, discipline",
    "finances, budgeting, debt, income goals, side projects",
    "hobbies, media, storytelling, creative projects",
    "AI and machine learning, building PRISM, technical concepts",
    "education, coursework, certifications, skill building",
    "daily routine, schedule optimization, productivity",
    "life goals, legacy, motivation, personal growth",
]


def get_claude_client():
    """Create an Anthropic client for generating user messages."""
    try:
        import anthropic
        api_key = config.MIRROR_ANTHROPIC_API_KEY or config.ANTHROPIC_API_KEY
        if not api_key:
            print("Error: No Anthropic API key configured.")
            sys.exit(1)
        return anthropic.Anthropic(api_key=api_key)
    except ImportError:
        print("Error: 'anthropic' package not installed.")
        sys.exit(1)


def load_context() -> str:
    """Load the user context for Claude to roleplay as."""
    return load_mirror_context()


def generate_user_message(
    client,
    context: str,
    conversation_history: list,
    topic: str | None,
    turn_number: int,
) -> str:
    """Ask Claude to generate a message as the user."""
    user_name = config.USER_NAME

    history_text = ""
    if conversation_history:
        recent = conversation_history[-6:]  # last 3 exchanges
        for msg in recent:
            role = user_name if msg["role"] == "user" else "PRISM"
            history_text += f"\n{role}: {msg['content']}\n"

    topic_instruction = ""
    if topic:
        topic_instruction = f"\nYou MUST focus this message on: {topic}. Do NOT drift to other topics."
    else:
        # Shuffle topics on first turn, then cycle through
        if turn_number == 1:
            random.shuffle(TOPICS)
        t = TOPICS[turn_number % len(TOPICS)]
        topic_instruction = f"\nYou MUST focus this message on: {t}. Stay on this topic. Do NOT default to relationship or geography talk unless that IS the topic."

    system = f"""You are roleplaying AS {user_name} in a conversation with their AI assistant PRISM.

USER CONTEXT:
{context}

RULES:
- Write exactly ONE message as {user_name} would say it
- Be natural, casual, conversational. Write how a real person texts
- Vary between short messages and longer ones
- Sometimes ask questions, sometimes share updates, sometimes test PRISM
- Reference real details from the user's life naturally
- Don't be robotic or overly structured
- Mix serious topics with casual chat
- Sometimes push back on generic responses
- Sometimes share emotions or frustrations
- Keep it under 3 sentences usually
- NO quotation marks around your message
- Do NOT prefix with "{user_name}:". Just write the message directly
{topic_instruction}"""

    user_prompt = f"Turn {turn_number}."
    if history_text:
        user_prompt = f"Recent conversation:{history_text}\n\nWrite {user_name}'s next message (turn {turn_number})."

    response = client.messages.create(
        model=config.MIRROR_ORACLE_MODEL,
        max_tokens=200,
        system=system,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text.strip()


def send_to_prism(base_url: str, session_id: str, message: str) -> dict:
    """Send a message to PRISM and get the response."""
    response = requests.post(
        f"{base_url}/chat",
        json={"session_id": session_id, "message": message},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def get_mirror_stats(base_url: str) -> dict:
    """Get current MIRROR convergence stats."""
    try:
        response = requests.get(f"{base_url}/mirror-status", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser(description="MIRROR Autopilot: Claude chats with PRISM as you")
    parser.add_argument("--turns", type=int, default=20, help="Number of conversation turns")
    parser.add_argument("--topic", type=str, default=None, help="Focus on a specific topic")
    parser.add_argument("--watch", action="store_true", help="Slow output for easier reading")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="PRISM server URL")
    parser.add_argument("--session", type=str, default="autopilot", help="Session ID")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  MIRROR Autopilot")
    print("  Claude is chatting with PRISM as you.")
    print(f"  Turns: {args.turns} | Topic: {args.topic or 'mixed'}")
    print("=" * 60 + "\n")

    # Verify PRISM is running
    try:
        requests.get(f"{args.url}/health", timeout=5)
    except Exception:
        print(f"Error: PRISM server not reachable at {args.url}")
        sys.exit(1)

    client = get_claude_client()
    context = load_context()
    conversation_history: list = []

    for turn in range(1, args.turns + 1):
        # Generate user message
        try:
            user_msg = generate_user_message(
                client, context, conversation_history, args.topic, turn
            )
        except Exception as e:
            print(f"\n  [Claude error: {e}]")
            continue

        # Send to PRISM
        try:
            result = send_to_prism(args.url, args.session, user_msg)
            prism_response = result["response"]
            episode_id = result["episode_id"]
        except Exception as e:
            print(f"\n  [PRISM error: {e}]")
            continue

        # Track conversation
        conversation_history.append({"role": "user", "content": user_msg})
        conversation_history.append({"role": "assistant", "content": prism_response})

        # Display
        print(f"  [{turn}/{args.turns}]")
        print(f"  {config.USER_NAME}: {user_msg}")
        print(f"  PRISM:   {prism_response[:200]}{'...' if len(prism_response) > 200 else ''}")
        print()

        if args.watch:
            time.sleep(3)
        else:
            time.sleep(0.5)  # small delay to let MIRROR score in background

    # Show final stats
    print("=" * 60)
    print("  Autopilot Complete")
    print(f"  Turns completed: {args.turns}")

    stats = get_mirror_stats(args.url)
    if stats:
        print(f"  Episodes auto-scored: {stats.get('episodes_auto_scored', 'N/A')}")
        avg = stats.get('rolling_avg_delta')
        print(f"  Rolling avg delta:    {avg:.3f}" if avg else "  Rolling avg delta:    N/A")
        print(f"  Mirror mode:          {stats.get('mirror_mode', 'N/A')}")

    print("=" * 60)
    print("\n  Run /train in the PRISM client to train on these episodes.")
    print("  Or wait for the 3 AM nightly run.\n")


if __name__ == "__main__":
    main()
