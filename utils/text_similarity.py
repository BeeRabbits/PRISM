"""Text similarity utilities. Trigram cosine, token F1, and a paraphrase-tolerant blend."""

import math
import re
from collections import Counter


_WORD_RE = re.compile(r"\b\w+\b")

# Minimal stopword list. Avoids penalizing paraphrases that swap filler words.
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "he", "i", "in", "is", "it", "its", "of", "on",
    "or", "she", "that", "the", "this", "to", "was", "were", "will",
    "with", "you", "your", "my", "me", "we", "us", "they", "them",
    "their", "so", "do", "does", "did", "but", "if", "then", "than",
    "just", "very", "can", "could", "would", "should", "been", "being",
})


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


def token_f1(text_a: str, text_b: str) -> float:
    """
    SQuAD-style token-level F1 on content words.

    Forgiving to word-order changes and filler differences, which is the
    common failure mode of trigram similarity when scoring paraphrases.
    """
    def tokens(t: str) -> list[str]:
        return [w for w in _WORD_RE.findall(t.lower()) if w not in _STOPWORDS]

    a, b = tokens(text_a), tokens(text_b)
    if not a or not b:
        return 0.0

    overlap = sum((Counter(a) & Counter(b)).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(a)
    recall = overlap / len(b)
    return 2 * precision * recall / (precision + recall)


def semantic_similarity(text_a: str, text_b: str) -> float:
    """
    Paraphrase-tolerant similarity: max(trigram cosine, token F1).

    Trigram catches character-level match; token-F1 catches word-level
    paraphrase. Taking the max means a strong signal from either one
    counts as "similar," reducing false-negatives on correct paraphrases.
    """
    return max(cosine_similarity_trigram(text_a, text_b), token_f1(text_a, text_b))
