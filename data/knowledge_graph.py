"""
KnowledgeGraph: CLS-inspired hippocampal index for PRISM.

Stores relational knowledge as (subject, predicate, object) triples in a
connected graph. Inspired by HippoRAG (NeurIPS 2024) but integrated with
PRISM's dream consolidation loop for continuous graph updates.

Architecture mapping:
    Brain                   PRISM
    ─────────────────────   ──────────────────────────
    Hippocampal index       This knowledge graph
    Pattern separation      Entity normalization
    Pattern completion      Graph traversal + PPR
    Systems consolidation   Dream consolidation → triple extraction

Key operations:
    1. add_triples()      — Insert new triples (called by dream consolidation)
    2. query_subgraph()   — Retrieve connected facts for inference injection
    3. personalized_pagerank() — Rank facts by relevance to query entities
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import config
from data.experience_log import Base, KnowledgeTripleRow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Entity normalization
# ---------------------------------------------------------------------------

# Common aliases that should resolve to the user entity
_USER_ALIASES = {
    "i", "me", "my", "myself", "the user", "user", "he", "she",
}

_STRIP_ARTICLES = {"the", "a", "an"}


def normalize_entity(entity: str, user_name: Optional[str] = None) -> str:
    """
    Normalize an entity string for consistent graph storage.

    - Lowercase and strip whitespace
    - Resolve user aliases to canonical user name
    - Strip leading articles
    """
    name = entity.strip().lower()

    # Strip leading articles
    for article in _STRIP_ARTICLES:
        if name.startswith(article + " "):
            name = name[len(article) + 1:]

    # Resolve user aliases
    uname = (user_name or config.USER_NAME).strip().lower()
    if name in _USER_ALIASES or name == uname:
        return uname

    return name


def normalize_predicate(predicate: str) -> str:
    """Normalize a predicate string: lowercase, strip, collapse whitespace."""
    return re.sub(r"\s+", "_", predicate.strip().lower())


# ---------------------------------------------------------------------------
# KnowledgeGraph
# ---------------------------------------------------------------------------

class KnowledgeGraph:
    """
    Async knowledge graph backed by SQLite.

    The graph stores (subject, predicate, object) triples with confidence
    scores and provenance tracking. Supports graph traversal and
    Personalized PageRank for relevance-weighted retrieval.

    Args:
        db_url: SQLAlchemy async database URL. Defaults to config.EXPERIENCE_DB_URL.
    """

    def __init__(self, db_url: Optional[str] = None) -> None:
        self._db_url = db_url or config.EXPERIENCE_DB_URL
        if self._db_url.startswith("sqlite"):
            db_path_str = self._db_url.split("///")[-1]
            Path(db_path_str).parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_async_engine(
            self._db_url,
            echo=False,
            connect_args={"timeout": 30},
        )
        self._session_factory = async_sessionmaker(
            self._engine, expire_on_commit=False, class_=AsyncSession
        )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 0.5,
        source_type: str = "consolidation",
        source_ids: Optional[List[str]] = None,
    ) -> str:
        """
        Add a knowledge triple to the graph.

        If an equivalent triple already exists (same subject, predicate, object),
        merges by incrementing validation_count and updating confidence.

        Returns:
            The triple's UUID string.
        """
        subj_norm = normalize_entity(subject)
        pred_norm = normalize_predicate(predicate)
        obj_norm = normalize_entity(obj)

        # Check for existing equivalent triple
        existing = await self._find_triple(subj_norm, pred_norm, obj_norm)
        if existing is not None:
            return await self._merge_triple(existing, confidence, source_ids or [])

        triple_id = str(uuid.uuid4())
        row = KnowledgeTripleRow(
            id=triple_id,
            subject=subj_norm,
            predicate=pred_norm,
            object=obj_norm,
            confidence=confidence,
            source_type=source_type,
            source_ids=json.dumps(source_ids or []),
            validation_count=1,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            is_active=True,
        )
        async with self._session_factory() as session:
            session.add(row)
            await session.commit()
        logger.debug("Added triple: (%s, %s, %s) [%s]", subj_norm, pred_norm, obj_norm, triple_id)
        return triple_id

    async def add_triples_batch(
        self,
        triples: List[dict],
        source_type: str = "consolidation",
        source_ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add multiple triples. Each dict should have keys: subject, predicate, object.
        Optional: confidence (default 0.5).

        Returns list of triple IDs.
        """
        ids = []
        for t in triples:
            tid = await self.add_triple(
                subject=t["subject"],
                predicate=t["predicate"],
                obj=t["object"],
                confidence=t.get("confidence", 0.5),
                source_type=source_type,
                source_ids=source_ids,
            )
            ids.append(tid)
        return ids

    async def deactivate_triple(self, triple_id: str) -> None:
        """Soft-delete a triple by marking it inactive."""
        async with self._session_factory() as session:
            await session.execute(
                update(KnowledgeTripleRow)
                .where(KnowledgeTripleRow.id == triple_id)
                .values(is_active=False, updated_at=datetime.utcnow())
            )
            await session.commit()

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def get_entity_triples(
        self,
        entity: str,
        hops: int = 1,
    ) -> List[dict]:
        """
        Retrieve all triples connected to an entity within N hops.

        Args:
            entity: The entity to start from.
            hops:   Number of graph hops (1 = direct connections, 2 = friends-of-friends).

        Returns:
            List of triple dicts.
        """
        entity_norm = normalize_entity(entity)
        visited_entities: Set[str] = set()
        result_triples: List[dict] = []
        frontier: Set[str] = {entity_norm}

        for _ in range(hops):
            if not frontier:
                break
            new_frontier: Set[str] = set()
            for ent in frontier:
                if ent in visited_entities:
                    continue
                visited_entities.add(ent)
                triples = await self._triples_for_entity(ent)
                for t in triples:
                    result_triples.append(t)
                    # Add connected entities to frontier for next hop
                    if t["subject"] not in visited_entities:
                        new_frontier.add(t["subject"])
                    if t["object"] not in visited_entities:
                        new_frontier.add(t["object"])
            frontier = new_frontier

        # Deduplicate by triple ID
        seen_ids: Set[str] = set()
        unique = []
        for t in result_triples:
            if t["id"] not in seen_ids:
                seen_ids.add(t["id"])
                unique.append(t)
        return unique

    async def query_subgraph(
        self,
        entities: List[str],
        hops: int = 2,
        top_k: int = 20,
    ) -> List[dict]:
        """
        Retrieve the connected subgraph for multiple query entities,
        ranked by Personalized PageRank.

        This is the main retrieval method called at inference time.

        Args:
            entities: List of entity strings extracted from user message.
            hops:     Graph traversal depth.
            top_k:    Maximum triples to return.

        Returns:
            List of triple dicts, ranked by relevance.
        """
        if not entities:
            return []

        # Gather all triples within hops of any query entity
        all_triples: List[dict] = []
        seen_ids: Set[str] = set()
        for entity in entities:
            triples = await self.get_entity_triples(entity, hops=hops)
            for t in triples:
                if t["id"] not in seen_ids:
                    seen_ids.add(t["id"])
                    all_triples.append(t)

        if not all_triples:
            return []

        # Rank by Personalized PageRank
        ranked = self._personalized_pagerank(all_triples, entities, top_k=top_k)
        return ranked

    async def get_all_entities(self) -> List[str]:
        """Return all unique entity names in the active graph."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(KnowledgeTripleRow.subject)
                .where(KnowledgeTripleRow.is_active == True)  # noqa: E712
                .distinct()
            )
            subjects = {row[0] for row in result.fetchall()}

            result = await session.execute(
                select(KnowledgeTripleRow.object)
                .where(KnowledgeTripleRow.is_active == True)  # noqa: E712
                .distinct()
            )
            objects = {row[0] for row in result.fetchall()}

        return sorted(subjects | objects)

    async def get_triple_count(self) -> int:
        """Return the number of active triples."""
        async with self._session_factory() as session:
            from sqlalchemy import func
            result = await session.execute(
                select(func.count(KnowledgeTripleRow.id))
                .where(KnowledgeTripleRow.is_active == True)  # noqa: E712
            )
            return result.scalar() or 0

    async def get_all_triples(self, limit: int = 500) -> List[dict]:
        """Return all active triples (for debugging/admin)."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(KnowledgeTripleRow)
                .where(KnowledgeTripleRow.is_active == True)  # noqa: E712
                .order_by(KnowledgeTripleRow.confidence.desc())
                .limit(limit)
            )
            rows = result.scalars().all()
        return [self._row_to_dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Entity extraction (lightweight, no API calls)
    # ------------------------------------------------------------------

    async def extract_entities_from_text(self, text: str) -> List[str]:
        """
        Extract known entities from text by matching against the graph's
        entity vocabulary. Zero API cost — pure dictionary lookup.

        Args:
            text: User message or other text to extract entities from.

        Returns:
            List of matched entity strings (normalized).
        """
        all_entities = await self.get_all_entities()
        if not all_entities:
            return []

        text_lower = text.lower()
        matched: List[str] = []

        # Sort by length (longest first) to match multi-word entities first
        for entity in sorted(all_entities, key=len, reverse=True):
            if entity in text_lower:
                matched.append(entity)
                # Remove matched entity to avoid sub-matches
                text_lower = text_lower.replace(entity, " ")

        # Also check user aliases
        user_name = config.USER_NAME.strip().lower()
        for alias in _USER_ALIASES:
            # Match whole words only
            if re.search(r"\b" + re.escape(alias) + r"\b", text.lower()):
                if user_name not in matched:
                    matched.append(user_name)
                break

        return matched

    # ------------------------------------------------------------------
    # Formatting for prompt injection
    # ------------------------------------------------------------------

    def format_for_injection(self, triples: List[dict]) -> str:
        """
        Format ranked triples as natural language for system prompt injection.

        Converts (subject, predicate, object) into readable sentences.
        """
        if not triples:
            return ""

        lines: List[str] = []
        for t in triples:
            subj = t["subject"].title()
            pred = t["predicate"].replace("_", " ")
            obj = t["object"].title() if not t["object"][0].isdigit() else t["object"]
            lines.append(f"- {subj} {pred} {obj}")

        return "Known facts about this user:\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Personalized PageRank
    # ------------------------------------------------------------------

    def _personalized_pagerank(
        self,
        triples: List[dict],
        query_entities: List[str],
        top_k: int = 20,
        damping: float = 0.85,
        iterations: int = 20,
    ) -> List[dict]:
        """
        Rank triples by Personalized PageRank relative to query entities.

        Builds an adjacency graph from triples, runs PPR with personalization
        vector concentrated on query entities, then scores each triple by
        the sum of its endpoint node scores.

        Args:
            triples:         All candidate triples.
            query_entities:  Entities to personalize toward.
            top_k:           Max triples to return.
            damping:         PPR damping factor (0.85 standard).
            iterations:      Number of power iterations.

        Returns:
            Top-k triples sorted by relevance score.
        """
        query_set = {normalize_entity(e) for e in query_entities}

        # Build adjacency list
        neighbors: Dict[str, Set[str]] = defaultdict(set)
        for t in triples:
            neighbors[t["subject"]].add(t["object"])
            neighbors[t["object"]].add(t["subject"])

        all_nodes = set(neighbors.keys())
        if not all_nodes:
            return triples[:top_k]

        # Initialize scores: personalization vector on query entities
        n = len(all_nodes)
        node_list = list(all_nodes)
        node_idx = {node: i for i, node in enumerate(node_list)}

        # Personalization vector: uniform over query entities
        personalization = [0.0] * n
        query_nodes_in_graph = [e for e in query_set if e in node_idx]
        if not query_nodes_in_graph:
            # No query entities found in graph, return by confidence
            return sorted(triples, key=lambda t: t["confidence"], reverse=True)[:top_k]

        for qe in query_nodes_in_graph:
            personalization[node_idx[qe]] = 1.0 / len(query_nodes_in_graph)

        # Power iteration
        scores = list(personalization)
        for _ in range(iterations):
            new_scores = [0.0] * n
            for i, node in enumerate(node_list):
                neighbor_set = neighbors.get(node, set())
                if not neighbor_set:
                    continue
                share = scores[i] / len(neighbor_set)
                for nb in neighbor_set:
                    if nb in node_idx:
                        new_scores[node_idx[nb]] += damping * share
            # Add personalization (teleport)
            for i in range(n):
                new_scores[i] += (1 - damping) * personalization[i]
            scores = new_scores

        # Score each triple by sum of endpoint scores
        triple_scores: List[Tuple[dict, float]] = []
        for t in triples:
            s_score = scores[node_idx[t["subject"]]] if t["subject"] in node_idx else 0
            o_score = scores[node_idx[t["object"]]] if t["object"] in node_idx else 0
            # Weight by confidence too
            relevance = (s_score + o_score) * t["confidence"]
            triple_scores.append((t, relevance))

        triple_scores.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in triple_scores[:top_k]]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _find_triple(
        self, subject: str, predicate: str, obj: str
    ) -> Optional[dict]:
        """Find an existing active triple with the same (s, p, o)."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(KnowledgeTripleRow)
                .where(KnowledgeTripleRow.subject == subject)
                .where(KnowledgeTripleRow.predicate == predicate)
                .where(KnowledgeTripleRow.object == obj)
                .where(KnowledgeTripleRow.is_active == True)  # noqa: E712
            )
            row = result.scalar_one_or_none()
        return self._row_to_dict(row) if row is not None else None

    async def _merge_triple(
        self, existing: dict, new_confidence: float, new_source_ids: List[str]
    ) -> str:
        """Merge a duplicate triple: bump validation count, update confidence."""
        merged_sources = list(set(existing.get("source_ids", []) + new_source_ids))
        # Confidence increases with validation (asymptotic toward 1.0)
        new_conf = min(1.0, existing["confidence"] + (1.0 - existing["confidence"]) * 0.2)

        async with self._session_factory() as session:
            await session.execute(
                update(KnowledgeTripleRow)
                .where(KnowledgeTripleRow.id == existing["id"])
                .values(
                    validation_count=existing["validation_count"] + 1,
                    confidence=new_conf,
                    source_ids=json.dumps(merged_sources),
                    updated_at=datetime.utcnow(),
                )
            )
            await session.commit()
        logger.debug(
            "Merged triple (%s, %s, %s): validation_count=%d, confidence=%.2f",
            existing["subject"], existing["predicate"], existing["object"],
            existing["validation_count"] + 1, new_conf,
        )
        return existing["id"]

    async def _triples_for_entity(self, entity: str) -> List[dict]:
        """Return all active triples where entity is subject or object."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(KnowledgeTripleRow)
                .where(KnowledgeTripleRow.is_active == True)  # noqa: E712
                .where(
                    (KnowledgeTripleRow.subject == entity)
                    | (KnowledgeTripleRow.object == entity)
                )
                .order_by(KnowledgeTripleRow.confidence.desc())
            )
            rows = result.scalars().all()
        return [self._row_to_dict(r) for r in rows]

    @staticmethod
    def _row_to_dict(row: KnowledgeTripleRow) -> dict:
        return {
            "id": row.id,
            "subject": row.subject,
            "predicate": row.predicate,
            "object": row.object,
            "confidence": row.confidence,
            "source_type": row.source_type,
            "source_ids": json.loads(row.source_ids or "[]"),
            "validation_count": row.validation_count,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
            "is_active": row.is_active,
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_graph_instance: Optional[KnowledgeGraph] = None


def get_knowledge_graph() -> KnowledgeGraph:
    """Return (or lazily create) the global KnowledgeGraph."""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = KnowledgeGraph()
    return _graph_instance
