# PRISM

**Persistent Reasoning and Intelligent Self-improving Model**

A locally-hosted LLM (Qwen2.5-14B-Instruct) that continuously evolves through interaction. PRISM combines continual fine-tuning, neural memory, oracle-bootstrapped self-evaluation, and neuroscience-inspired training into a single system where the model's weights shift toward what matters over time.

This is **not** RAG. This is **not** external memory injection. The model itself changes.

---

## Core Innovations

### MIRROR Protocol: Oracle-Bootstrapped Self-Evaluation

MIRROR solves a fundamental problem: how does a model learn to judge its own response quality without constant human feedback?

1. For every conversation, an Oracle (Claude) independently scores the response alongside PRISM's own self-score.
2. The **delta** between Oracle and self-score is tracked over a rolling window.
3. High-delta episodes (where PRISM misjudged quality the most) are **overweighted** in training, up to 3x repetition.
4. As the rolling average delta drops below the convergence threshold (0.30), PRISM's judgment has aligned with the Oracle's.
5. Once converged, Oracle calls stop entirely. PRISM scores autonomously.

This is essentially **bootstrapped alignment for self-evaluation**. The stronger model teaches the weaker model to judge itself, then steps away.

See [MIRROR.md](MIRROR.md) for the full protocol specification.

### Neuroscience-Inspired Training Pipeline

PRISM's training is modeled on how the human brain consolidates and retrieves memory:

**Spaced Repetition:** Episodes are replayed at Fibonacci intervals (1, 1, 2, 3, 5, 8, 13, 21, 34, 55 training cycles). Memories reviewed at increasing intervals are retained far longer than massed repetition.

**Active Recall:** Before each training run, the model is tested on previously trained episodes. Episodes where recall falls below threshold are re-queued. Testing strengthens retention more than passive re-reading.

**TMR Dream Consolidation:** Targeted Memory Reactivation. Episodes with the highest delta between original and Oracle scores are over-represented in training batches. High-salience memories get extra consolidation passes, mirroring what happens during human sleep.

**Interleaved Training:** Topics are ordered in round-robin rather than sequential blocks. Interleaving causes short-term difficulty but produces better long-term retention and transfer, a well-established finding in human learning research.

**Mixed Knowledge (60/30/10 Split):** Every training batch is composed of three sources to prevent catastrophic forgetting:

| Proportion | Source |
|---|---|
| 60% | High-fitness episodic memories (user interactions) |
| 30% | General knowledge Q&A (preserves world knowledge) |
| 10% | Spaced-replay episodes (Fibonacci-scheduled reruns) |

### Dream Consolidation

Dream Consolidation compresses clusters of related episodic memories into validated **semantic memories**, distilled truths about the user for future training.

- **Cluster**: groups similar episodes by cosine similarity on character trigrams
- **Consolidate**: asks Claude to compress each cluster into one semantic memory
- **Safeguard**: rejects semantics with LLM confidence < 0.80 or where the cluster is too varied
- **Validate**: provisional semantics must be reinforced by 3+ matching recent episodes before promotion
- **Prune**: provisional semantics older than 14 days with zero validations are deleted

Semantic memories are never injected from the start. They must **earn confirmation** through real usage patterns.

### Contradiction Engine

Before every episode is logged, it is compared against the user's recent history. If high similarity is detected, Claude evaluates whether a genuine contradiction exists.

Resolutions: `keep_a` (existing wins), `keep_b` (new wins), `merge` (combine both), or `keep_both` (no real contradiction). This prevents the model from training on conflicting information.

### Cortex Loop

Post-convergence, Cortex Loop increases reasoning depth by duplicating transformer layers at an optimal "seam" point in the network. Inspired by the [RYS (Repeat Your Self)](https://github.com/dnhkng/RYS) technique. A scan sweeps candidate positions and layer counts, scoring each on reasoning benchmarks before applying the best configuration.

---

## Architecture

### Dual-Speed Memory System

| Component | Speed | Persistence | What it stores |
|---|---|---|---|
| **QLoRA adapters** (about 50 MB) | Slow | Permanent weight changes | Values, style, preferences, knowledge |
| **Titans memory adapter** | Fast | Per-session, evolves via training | In-context patterns, conversation state |

**QLoRA**: The base model (29 GB) is loaded in 4-bit NF4 quantization. Only small adapter matrices are trained, achieving full fine-tuning quality at 1/4 the memory cost.

**Titans adapter**: Standard transformer attention resets each call. The Titans adapter maintains a persistent memory bank across turns within a session, grafted at the embedding layer as a hook. The base model is never modified directly.

### Why Not RAG?

RAG is retrieval. You fetch documents and inject them as context. The model doesn't *know* anything. PRISM makes the model *learn*. The weights shift toward the patterns that mattered. The knowledge is structural, not retrieved.

---

## Training Loop

1. Every `/chat` call logs an episode. The Contradiction Engine screens it first.
2. Every `/feedback` call updates that episode's fitness score (0.0 to 1.0).
3. At 3 AM daily (configurable):
   1. Dream Consolidation clusters and compresses episodic memories
   2. DatasetBuilder assembles the 60/30/10 training batch with interleaved topic ordering
   3. LoRA training runs on the combined dataset
   4. Titans adapter trains on the same data
   5. New adapters are hot-swapped into the live model (no restart needed)
4. Tomorrow the model is slightly better.

---

## Training Results

Across multiple training cycles on 2,500+ episodes:

| Run | LoRA Eval Loss | Titans Loss | General Knowledge % |
|---|---|---|---|
| 1 | 0.788 | 0.612 | 1% (71 examples) |
| 2 | 0.789 | 0.614 | 9% (571 examples) |
| 3 | 0.784 | 0.612 | 9% (571 examples) |
| 4 (reset) | 0.821 | 0.628 | 9% (571 examples) |

MIRROR delta: 2.145 to 1.900 over training cycles (convergence target: 0.30).

Early training with only 1% general knowledge caused catastrophic forgetting (model output degraded to Chinese text and gibberish). Expanding to 9% restored coherence. The dataset now contains 2,456 general knowledge entries targeting the full 30% ratio.

---

## Quick Start

**Requirements:** Ubuntu 22/24, NVIDIA GPU (A100 40GB recommended), CUDA 12.4+

```bash
# Clone and setup
git clone https://github.com/BeeRabbits/PRISM.git
cd PRISM
bash setup.sh

# Configure
cp .env.example .env
# Edit .env: set HF_TOKEN, ANTHROPIC_API_KEY (for MIRROR + Dream Consolidation)

# Download model (about 29 GB)
source .venv/bin/activate
python scripts/download_model.py

# Start
python main.py
```

### Chat

```bash
python prism_inside_client.py http://localhost:8000 my_session
```

### MIRROR Autopilot

Claude simulates conversations as the user, PRISM responds, and MIRROR auto-scores each exchange:

```bash
python scripts/mirror_autopilot.py --turns 500
```

---

## Tech Stack

- **Base model**: Qwen2.5-14B-Instruct (4-bit NF4 quantization)
- **Fine-tuning**: QLoRA via PEFT + SFTTrainer
- **Memory**: Custom Titans cross-attention adapter with MaG gating
- **Oracle**: Claude (Anthropic API) for MIRROR scoring + Dream Consolidation
- **Server**: FastAPI + Uvicorn
- **Storage**: SQLite (episodes, semantic memories, contradiction logs)
- **Training**: PyTorch + Transformers + BitsAndBytes

---

## Acknowledgments and References

PRISM builds on the work of many researchers and open-source projects:

**Models and Frameworks**
- [Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) by Alibaba Qwen Team
- [Claude](https://www.anthropic.com/) by Anthropic, used as the MIRROR Oracle and for Dream Consolidation
- [PyTorch](https://pytorch.org/), [Transformers](https://huggingface.co/docs/transformers), [PEFT](https://huggingface.co/docs/peft), and [TRL](https://huggingface.co/docs/trl) by Hugging Face
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) by Tim Dettmers

**Research Papers**
- **Titans adapter**: "Titans: Learning to Memorize at Test Time" (Google DeepMind, 2024). PRISM's memory adapter and MaG gating mechanism are based on this work.
- **LoRA**: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., Microsoft Research, 2021)
- **QLoRA**: "QLoRA: Efficient Finetuning of Quantized Language Models" (Dettmers et al., 2023)
- **RYS**: [Repeat Your Self](https://github.com/dnhkng/RYS) by dnhkng. Cortex Loop's layer duplication technique is inspired by this project.

**Neuroscience Foundations**
- **Spaced Repetition**: Based on the spacing effect (Ebbinghaus, 1885) and modern spaced repetition research (Piotr Wozniak)
- **Active Recall**: "Test-Enhanced Learning" (Roediger and Karpicke, 2006). Testing strengthens retention more than re-study.
- **Targeted Memory Reactivation (TMR)**: Research by Rasch et al. (2007) and Oudiette and Paller (2013) on memory reactivation during sleep
- **Interleaved Practice**: Research by Rohrer and Taylor (2007) and Kornell and Bjork (2008) on the benefits of interleaved over blocked practice
- **Catastrophic Forgetting**: McCloskey and Cohen (1989). The mixed knowledge dataset is designed to prevent this well-documented failure mode in continual learning.

**Built with**
- [Claude Code](https://claude.ai/code) by Anthropic

---

## License

MIT
