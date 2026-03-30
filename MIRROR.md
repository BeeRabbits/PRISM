# MIRROR Protocol

**Model Iteratively Refined by Recursive Oracle Referencing**

## Overview

MIRROR is a novel training architecture that eliminates manual feedback scoring
by combining two automated signals:

1. **Oracle Distillation**. Claude evaluates every PRISM response against a
   personalized user context file, producing a quality score (1-5).

2. **Self-Referential Loop**. PRISM evaluates its own responses using local
   inference. The **delta** between PRISM's self-score and Claude's oracle score
   becomes the primary training signal.

High delta = PRISM misjudged its own quality = high learning priority.

## How It Works

```
User sends message
        │
        ▼
PRISM generates response ──────────────────► Response sent to user
        │                                    (no latency added)
        ▼
   [Background]
        │
   ┌────┴────┐
   │         │
Oracle    Self-Score
(Claude)  (PRISM local)
   │         │
   └────┬────┘
        │
    Compute Delta
   |oracle - self|
        │
        ▼
  Update Episode
  (score + delta)
        │
        ▼
  Nightly Training
  (delta-weighted)
```

## Convergence

Over time, PRISM's self-scores converge toward Claude's oracle scores, meaning
PRISM has internalized the user's quality standards. When the rolling average
delta drops below 0.30 over 50 episodes:

- `MIRROR_CONVERGED` is set to true
- Claude oracle calls stop (saves API cost)
- PRISM runs fully autonomously using its own self-scoring

## Training Integration

Episodes are weighted in LoRA training based on their delta:

- **High delta (>2)**: 3x weight (PRISM was most wrong, highest learning value)
- **Medium delta (1-2)**: 2x weight
- **Low delta (<1)**: 1x weight (already aligned)
- **No delta (legacy/manual)**: 1x weight

## Why This Matters

MIRROR addresses a core challenge in personalized AI: how do you train a model to align with a specific user's preferences without requiring that user to manually score every response?

By using a stronger model (Claude) as a temporary oracle, PRISM bootstraps its own evaluation ability. The oracle is not permanent. It is a scaffold that gets removed once the model can stand on its own. This pattern could generalize to any scenario where a capable teacher model helps a smaller model learn self-assessment, reducing long-term dependency on expensive external scoring.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MIRROR_ENABLED` | `true` | Enable/disable MIRROR |
| `MIRROR_ORACLE_MODEL` | `claude-sonnet-4-6` | Claude model for oracle |
| `MIRROR_ANTHROPIC_API_KEY` | (falls back to ANTHROPIC_API_KEY) | API key for oracle |
| `MIRROR_CONTEXT_FILE` | `./data/mirror_context.md` | User context file path |
| `MIRROR_CONVERGENCE_THRESHOLD` | `0.30` | Delta threshold for convergence |
| `MIRROR_CONVERGENCE_WINDOW` | `50` | Episodes to average for convergence |
| `MIRROR_CONVERGED` | `false` | Set true when converged |

## Commands

- `/mirror` in the chat client shows convergence status
- `GET /mirror-status` returns full metrics via API
- `python scripts/mirror_bootstrap.py <file>` imports prior conversations

## Current Progress

| Metric | Value |
|--------|-------|
| Episodes scored | 2,500+ |
| Starting delta | 2.145 |
| Current delta | 1.900 |
| Convergence target | 0.30 |
| Training runs | 4 |

The delta is actively decreasing with each training cycle, indicating the model is learning to self-evaluate more accurately over time.
