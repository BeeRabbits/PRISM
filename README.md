# PRISM

**Persistent Reasoning and Intelligent Self-improving Model**

A locally-hosted LLM (Qwen2.5-32B-Instruct, 4-bit QLoRA, runs on a single 48 GB Ada GPU such as the RTX 6000 Ada) that continuously evolves through interaction. PRISM combines continual fine-tuning, neural memory, oracle-bootstrapped self-evaluation, a hippocampal knowledge graph, and neuroscience-inspired training into a single system where the model's weights shift toward what matters over time.

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

### Knowledge Graph: Hippocampal Indexing

Inspired by HippoRAG, PRISM maintains a knowledge graph of (subject, predicate, object) triples extracted from consolidated memories. At inference time, entities are extracted from the user's message, expanded via query expansion with stemming, and relevant subgraphs are retrieved using Personalized PageRank with keyword relevance boosting.

This acts as the hippocampal index — fast, associative retrieval that feeds structured facts into the system prompt. Combined with weight-level learning, PRISM both *knows* things (weights) and can *look up* things (graph).

- **Entity normalization**: User name variants, stemming, article stripping
- **Query expansion**: 1-hop predicate/object matching discovers related entities
- **Keyword boost**: 3x score multiplier for triples whose predicate/object matches query terms
- **Personalized PageRank**: Ranks triples by relevance to the query context

### Evaluation Gates

Automated pre/post training evaluation prevents catastrophic forgetting at the system level. A test suite of general capability tests (code debugging, recipe generation, email writing) and personal recall tests (user facts, preferences, relationships) runs before and after every training cycle.

- General capability must stay >= 80% or the adapter is rolled back
- Personal recall is tracked but informational (retrieval handles most of it)
- Baseline comparison detects regression > 10%

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

### MoE-LoRA Expert Router

Multiple specialized LoRA adapters can be trained for different domains (e.g., a "style" expert for conversational tone). A keyword-based router activates the appropriate expert at inference time. Each expert is trained with its own evaluation gate to prevent catastrophic forgetting.

### Cortex Loop

Post-convergence, Cortex Loop increases reasoning depth by duplicating transformer layers at an optimal "seam" point in the network. Inspired by the [RYS (Repeat Your Self)](https://github.com/dnhkng/RYS) technique. A scan sweeps candidate positions and layer counts, scoring each on reasoning benchmarks before applying the best configuration.

### Runtime Adaptations

**Frustration Detection:** A 17-pattern regex system detects user sentiment (NONE, MILD, FRUSTRATED, ANGRY) before every inference call at zero API cost. When frustration is detected, the system prompt is dynamically modified to adjust tone, acknowledge difficulty, and offer more direct help.

**Idle-Time Consolidation:** When the server detects prolonged inactivity (default: 30 minutes), it automatically triggers Dream Consolidation. This mirrors the brain's tendency to consolidate memories during rest, making use of idle GPU time that would otherwise be wasted.

**Context Compaction:** Server-side session history tracks conversation turns per session. When the estimated token count exceeds a threshold, older messages are summarized by the local model and replaced with a compact summary, keeping recent turns intact. This prevents context degradation during long conversations without any API cost.

**Memory Validation:** Semantic memories are validated before injection into inference. Each memory is checked against source episode fitness, staleness (with a 0.7x confidence penalty for stale memories), and minimum confidence thresholds. Memories must earn their way into the prompt rather than being blindly trusted.

---

## Architecture

### Dual-Speed Memory System

| Component | Speed | Persistence | What it stores |
|---|---|---|---|
| **QLoRA adapters** (~17 MB) | Slow | Permanent weight changes | Values, style, preferences, knowledge |
| **Titans memory adapter** | Fast | Per-session, evolves via training | In-context patterns, conversation state |
| **Knowledge graph** | Instant | Persistent triple store | Structured facts, relationships, entities |

**QLoRA**: The base model is loaded in 4-bit NF4 quantization. Only small adapter matrices are trained, achieving full fine-tuning quality at a fraction of the memory cost.

**Titans adapter**: Standard transformer attention resets each call. The Titans adapter maintains a persistent memory bank across turns within a session, grafted at the embedding layer as a hook. The base model is never modified directly.

**Knowledge graph**: HippoRAG-inspired triple store with entity normalization, query expansion, and Personalized PageRank retrieval. Facts are extracted from consolidated semantic memories and injected into the system prompt at inference time.

### Why Not RAG?

RAG is retrieval. You fetch documents and inject them as context. The model doesn't *know* anything. PRISM makes the model *learn*. The weights shift toward the patterns that mattered. The knowledge is structural, not retrieved.

The knowledge graph complements weight-level learning — it handles factual recall (names, dates, relationships) while the weights encode style, preferences, and deeper understanding.

---

## Training Loop

1. Every `/chat` call logs an episode. The Contradiction Engine screens it first.
2. Every `/feedback` call updates that episode's fitness score (0.0 to 1.0).
3. MIRROR Oracle (Claude) auto-scores each episode in the background.
4. At 3 AM daily (configurable):
   1. Active Recall tests retention on 30 previously trained episodes
   2. Dream Consolidation clusters and compresses episodic memories into semantics
   3. Knowledge graph triples are extracted from new semantic clusters
   4. DatasetBuilder assembles the 60/30/10 training batch with interleaved topic ordering
   5. Evaluation gate runs pre-training baseline
   6. LoRA training runs on the combined dataset
   7. Titans adapter trains on the same data
   8. Post-training evaluation checks for regression
   9. New adapters are hot-swapped into the live model (no restart needed)
5. Tomorrow the model is slightly better.

---

## Training Results

Latest training on 2,900+ episodes with Qwen2.5-32B-Instruct:

| Metric | Value |
|---|---|
| LoRA train loss | 1.74 → 1.10 |
| LoRA eval loss | 1.128 |
| Titans avg loss | 1.276 |
| Trainable params (legacy q/v rank-8) | 8.4M / 32.8B (0.026%) |
| Trainable params (current wide rank-32) | ~55M / 32.8B (~0.17%) |
| MIRROR rolling avg delta | 1.660 |
| Avg oracle score (32B) | 2.3 / 5.0 |
| General eval | 100% (10/10) |
| Personal eval | 100% (10/10) |
| Knowledge graph | 402 triples, 327 entities |

The 60/30/10 mixed training split (episodic, general knowledge, spaced replay) successfully prevents catastrophic forgetting while allowing the model to learn from user interactions.

**Adapter capacity upgrade.** The default LoRA config now targets all 7 linear modules (`q,k,v,o,gate,up,down`) at rank 32 instead of the legacy `q_proj,v_proj` rank-8 setup. That moves trainable params from ~8.4M (0.026%) to ~55M (~0.17%) — roughly 6.5× more adapter capacity, on the same hardware, in the same per-cycle training time. This is the standard QLoRA recipe (Dettmers et al., 2023) and is the single biggest lever for personalization quality before changing the base model. To revert to the narrow legacy config, set `LORA_RANK=8` and `LORA_TARGET_MODULES=q_proj,v_proj` in `.env`.

---

## Quick Start

**Requirements:** Ubuntu 22/24, NVIDIA GPU with **48GB+ VRAM** (Ada Lovelace or newer recommended), CUDA 12.4+

**Recommended pod (RunPod):** **RTX 6000 Ada (48 GB) @ ~$0.77/hr** — Ada Lovelace gives ~2.5–3× faster training cycles than the older A40 at the same VRAM, with FlashAttention-2 support and 62 GB system RAM (no CPU-RAM cliff at model load). $100 of credit ≈ 130 hours of training.

| Tier | GPU | VRAM | $/hr (RunPod) | Notes |
|---|---|---|---|---|
| **Recommended** | RTX 6000 Ada | 48 GB | $0.77 | Best $/training-speed for 32B QLoRA |
| Budget | A40 | 48 GB | $0.44 ($0.22 spot) | Works but ~3× slower; 50 GB RAM tight on load |
| Speed | L40S | 48 GB | $0.86 ($0.71 spot) | Same Ada gen, slightly faster |
| Bursts | H100 PCIe | 80 GB | $2.03 spot | Fastest wall-clock; ~10–15 min cycles |
| Headroom | RTX PRO 6000 | 96 GB | $1.61 spot | Blackwell, room to grow LoRA rank further |

PRISM is single-GPU by design (no FSDP/DDP). 2× GPU pods (e.g. 2× RTX 5090) provide no training speedup here.

```bash
# Clone
git clone https://github.com/BeeRabbits/PRISM.git
cd PRISM

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env: set HF_TOKEN, ANTHROPIC_API_KEY, USER_NAME

# Download model
python scripts/download_model.py

# Start
python main.py
```

For a full fresh Ubuntu setup (system packages, venv, CUDA verification), use `bash setup.sh` instead.

### Chat

```bash
python prism_inside_client.py http://localhost:8000 my_session
```

### MIRROR Autopilot

Claude simulates conversations as the user, PRISM responds, and MIRROR auto-scores each exchange:

```bash
python scripts/mirror_autopilot.py --turns 500
```

### Data Management

```bash
# Clean corrupted episodes (Chinese characters, underscore artifacts)
python scripts/clean_episodes.py

# Seed knowledge graph from semantic memories
python scripts/seed_knowledge_graph.py

# Merge duplicate knowledge graph entities
python scripts/merge_kg_entities.py
```

---

## Tech Stack

- **Base model**: Qwen2.5-32B-Instruct (4-bit NF4 quantization)
- **Fine-tuning**: QLoRA via PEFT + SFTTrainer (rank 32 across all 7 linear modules, ~55M trainable params)
- **Attention kernel**: FlashAttention-2 on Ada/Hopper (auto-falls-back to SDPA)
- **Memory**: Custom Titans cross-attention adapter with MaG gating
- **Knowledge**: HippoRAG-inspired triple store with Personalized PageRank
- **Oracle**: Claude (Anthropic API) for MIRROR scoring + Dream Consolidation
- **Server**: FastAPI + Uvicorn
- **Storage**: SQLite (episodes, semantic memories, contradiction logs, knowledge triples)
- **Training**: PyTorch + Transformers + BitsAndBytes
- **Evaluation**: Automated test suite with regression detection and auto-rollback

---

## Acknowledgments and References

PRISM builds on the work of many researchers and open-source projects:

**Models and Frameworks**
- [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) by Alibaba Qwen Team
- [Claude](https://www.anthropic.com/) by Anthropic, used as the MIRROR Oracle and for Dream Consolidation
- [PyTorch](https://pytorch.org/), [Transformers](https://huggingface.co/docs/transformers), [PEFT](https://huggingface.co/docs/peft), and [TRL](https://huggingface.co/docs/trl) by Hugging Face
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) by Tim Dettmers

**Research Papers**
- **Titans adapter**: "Titans: Learning to Memorize at Test Time" (Google DeepMind, 2024). PRISM's memory adapter and MaG gating mechanism are based on this work.
- **LoRA**: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., Microsoft Research, 2021)
- **QLoRA**: "QLoRA: Efficient Finetuning of Quantized Language Models" (Dettmers et al., 2023)
- **HippoRAG**: "HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models" (Gutierrez et al., 2024). PRISM's knowledge graph architecture is inspired by this work.
- **RYS**: [Repeat Your Self](https://github.com/dnhkng/RYS) by dnhkng. Cortex Loop's layer duplication technique is inspired by this project.

**Neuroscience Foundations**
- **Complementary Learning Systems (CLS)**: McClelland, McNaughton, and O'Reilly (1995). The dual-speed architecture (fast hippocampal KG + slow neocortical weights) is based on CLS theory.
- **Spaced Repetition**: Based on the spacing effect (Ebbinghaus, 1885) and modern spaced repetition research (Piotr Wozniak)
- **Active Recall**: "Test-Enhanced Learning" (Roediger and Karpicke, 2006). Testing strengthens retention more than re-study.
- **Targeted Memory Reactivation (TMR)**: Research by Rasch et al. (2007) and Oudiette and Paller (2013) on memory reactivation during sleep
- **Interleaved Practice**: Research by Rohrer and Taylor (2007) and Kornell and Bjork (2008) on the benefits of interleaved over blocked practice
- **Catastrophic Forgetting**: McCloskey and Cohen (1989). The mixed knowledge dataset is designed to prevent this well-documented failure mode in continual learning.

**Built with**
- [Claude Code](https://claude.ai/code) by Anthropic

---

## License

[Business Source License 1.1](LICENSE) - Free for non-commercial and educational use. Converts to Apache 2.0 on 2030-04-01.
