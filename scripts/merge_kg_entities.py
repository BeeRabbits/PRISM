"""
One-time script: merge fragmented user entities in the knowledge graph.

The initial seeding created triples with "user", "brandon", "brandon peffer"
as separate entities. This script normalizes them all to the canonical
USER_NAME from config.

Usage:
    python scripts/merge_kg_entities.py
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.knowledge_graph import get_knowledge_graph, normalize_entity

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s")
logger = logging.getLogger(__name__)


async def merge_user_entities():
    """Re-normalize all subject and object fields in the knowledge graph."""
    from sqlalchemy import select, update, text
    from data.experience_log import KnowledgeTripleRow

    kg = get_knowledge_graph()
    uname = config.USER_NAME.strip().lower()

    # Variants that should all map to the canonical user name
    user_variants = {
        "user", "the user", "brandon", "brandon peffer",
        "i", "me", "my", "myself", "he", "she",
    }
    user_variants.add(uname)

    logger.info("Canonical user name: '%s'", uname)
    logger.info("Merging these variants: %s", user_variants)

    async with kg._session_factory() as session:
        # Get all triples
        result = await session.execute(
            select(KnowledgeTripleRow)
            .where(KnowledgeTripleRow.is_active == True)  # noqa: E712
        )
        rows = result.scalars().all()
        logger.info("Total active triples: %d", len(rows))

        updated = 0
        for row in rows:
            new_subj = normalize_entity(row.subject)
            new_obj = normalize_entity(row.object)
            new_pred = row.predicate  # already normalized

            changed = False
            if new_subj != row.subject:
                changed = True
            if new_obj != row.object:
                changed = True

            if changed:
                await session.execute(
                    update(KnowledgeTripleRow)
                    .where(KnowledgeTripleRow.id == row.id)
                    .values(subject=new_subj, object=new_obj)
                )
                updated += 1

        await session.commit()
        logger.info("Updated %d triples with normalized entities.", updated)

    # Now merge exact duplicates that were created by the normalization
    await _merge_duplicates(kg)


async def _merge_duplicates(kg):
    """Find and merge triples that are now exact duplicates after normalization."""
    from sqlalchemy import select, delete, func
    from data.experience_log import KnowledgeTripleRow

    async with kg._session_factory() as session:
        # Find groups with same (subject, predicate, object) that have multiple rows
        result = await session.execute(
            select(
                KnowledgeTripleRow.subject,
                KnowledgeTripleRow.predicate,
                KnowledgeTripleRow.object,
                func.count(KnowledgeTripleRow.id).label("cnt"),
            )
            .where(KnowledgeTripleRow.is_active == True)  # noqa: E712
            .group_by(
                KnowledgeTripleRow.subject,
                KnowledgeTripleRow.predicate,
                KnowledgeTripleRow.object,
            )
            .having(func.count(KnowledgeTripleRow.id) > 1)
        )
        dupes = result.fetchall()
        logger.info("Found %d duplicate triple groups to merge.", len(dupes))

        merged_count = 0
        for subj, pred, obj, cnt in dupes:
            # Get all rows for this triple
            rows_result = await session.execute(
                select(KnowledgeTripleRow)
                .where(KnowledgeTripleRow.subject == subj)
                .where(KnowledgeTripleRow.predicate == pred)
                .where(KnowledgeTripleRow.object == obj)
                .where(KnowledgeTripleRow.is_active == True)  # noqa: E712
                .order_by(KnowledgeTripleRow.created_at.asc())
            )
            rows = rows_result.scalars().all()

            # Keep the first (oldest), merge validation counts, delete rest
            keeper = rows[0]
            total_validations = sum(r.validation_count for r in rows)
            best_confidence = max(r.confidence for r in rows)

            from sqlalchemy import update
            await session.execute(
                update(KnowledgeTripleRow)
                .where(KnowledgeTripleRow.id == keeper.id)
                .values(
                    validation_count=total_validations,
                    confidence=best_confidence,
                )
            )

            # Deactivate duplicates
            for row in rows[1:]:
                await session.execute(
                    update(KnowledgeTripleRow)
                    .where(KnowledgeTripleRow.id == row.id)
                    .values(is_active=False)
                )
                merged_count += 1

        await session.commit()
        logger.info("Merged %d duplicate triples.", merged_count)

    # Final stats
    count = await kg.get_triple_count()
    entities = await kg.get_all_entities()
    logger.info(
        "After merge: %d active triples, %d unique entities.",
        count, len(entities),
    )


async def main():
    from data.experience_log import get_experience_logger
    exp_log = get_experience_logger()
    await exp_log.init()

    await merge_user_entities()


if __name__ == "__main__":
    asyncio.run(main())
