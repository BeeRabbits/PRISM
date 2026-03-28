"""Trigram cosine similarity. Shared utility for text comparison."""

import math
import re
from collections import Counter


def cosine_similarity_trigram(text_a: str, text_b: str) -> float:
    """
    Compute cosine similarity between two texts using character trigrams.
    Fast, dependency-free, and adequate for same-domain episode clustering.
    """
    def trigrams(text: str) -> Counter:
        t = re.sub(r"\s+", " ", text.lower().strip())
        return Counter(t[i : i + 3] for i in range(len(t) - 2))

    v1, v2 = trigrams(text_a), trigrams(text_b)
    if not v1 or not v2:
        return 0.0

    common = set(v1) & set(v2)
    dot = sum(v1[k] * v2[k] for k in common)
    mag1 = math.sqrt(sum(v ** 2 for v in v1.values()))
    mag2 = math.sqrt(sum(v ** 2 for v in v2.values()))

    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)
