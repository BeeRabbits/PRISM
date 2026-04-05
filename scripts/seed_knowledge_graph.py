"""
One-time script: extract knowledge triples from existing semantic memories
and seed the knowledge graph.

Run once to bootstrap the graph from memories created before the KG existed.
After this, new triples are extracted automatically during dream consolidation.

Usage:
    python scripts/seed_knowledge_graph.py
"""

import asyncio
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.knowledge_graph import get_knowledge_graph
from data.semantic_store import get_semantic_store
from data.experience_log import get_experience_logger

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s")
logger = logging.getLogger(__name__)


async def extract_triples_from_semantics():
    """Use Claude to extract triples from all existing semantic memories."""
    import anthropic

    store = get_semantic_store()
    kg = get_knowledge_graph()

    # Get all semantic memories (confirmed + provisional)
    confirmed = await store.get_confirmed_semantics(limit=500)
    provisional = await store.get_provisional_semantics()
    all_semantics = confirmed + provisional

    if not all_semantics:
        logger.info("No semantic memories found. Nothing to seed.")
        return

    logger.info("Seeding knowledge graph from %d semantic memories...", len(all_semantics))

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    # Batch semantics into groups of 10 to reduce API calls
    batch_size = 10
    total_triples = 0

    for i in range(0, len(all_semantics), batch_size):
        batch = all_semantics[i:i + batch_size]
        formatted = "\n\n".join(
            f"Memory {j+1}: {sem['content']}\n(Pattern: {sem['key_pattern']})"
            for j, sem in enumerate(batch)
        )

        prompt = f"""Extract factual knowledge as (subject, predicate, object) triples from these memories about a user.

Rules:
- Use the person's actual name as subject when known, not "the user"
- Each triple should be a single clear fact
- Predicates should be short verb phrases (e.g., "works at", "has son named", "interested in")
- Be specific: "Brandon" not "user", "Landen" not "his son"
- Include relationships, facts, preferences, goals, dates, locations

Memories:
{formatted}

Return a JSON array of triples only:
[{{"subject": "...", "predicate": "...", "object": "..."}}, ...]"""

        try:
            response = client.messages.create(
                model=config.ANTHROPIC_MODEL,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            # Strip markdown fences
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            triples = json.loads(raw)

            if triples:
                source_ids = [sem["id"] for sem in batch]
                added = await kg.add_triples_batch(
                    triples,
                    source_type="consolidation",
                    source_ids=source_ids,
                )
                total_triples += len(added)
                logger.info(
                    "Batch %d-%d: extracted %d triples",
                    i + 1, min(i + batch_size, len(all_semantics)), len(added),
                )

                # Print sample
                for t in triples[:3]:
                    logger.info("  (%s, %s, %s)", t["subject"], t["predicate"], t["object"])

        except json.JSONDecodeError as e:
            logger.error("JSON parse error on batch %d: %s", i, e)
        except Exception as e:
            logger.error("API error on batch %d: %s", i, e)

    # Also seed from episode data directly (high-fitness episodes)
    exp_log = get_experience_logger()
    await exp_log.init()

    count = await kg.get_triple_count()
    entities = await kg.get_all_entities()
    logger.info(
        "Knowledge graph seeded: %d total triples, %d unique entities.",
        count, len(entities),
    )
    logger.info("Entities: %s", ", ".join(entities[:30]))


async def main():
    # Initialize DB tables (ensures knowledge_triples table exists)
    exp_log = get_experience_logger()
    await exp_log.init()

    await extract_triples_from_semantics()


if __name__ == "__main__":
    asyncio.run(main())
