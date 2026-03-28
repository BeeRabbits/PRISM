# PRISM

**Persistent Resonance Intelligence & Self-Modifying Memory**

A locally-hosted LLM (Qwen2.5-14B-Instruct) that continuously evolves through:

1. **QLoRA continual fine-tuning**: slow weight drift. Values, style, and preferences baked into the model over time.
2. **Grafted Titans memory adapter**: fast in-context neural memory. Compresses noisy context into clean signals at the embedding layer.
3. **MIRROR Protocol**: Oracle-guided auto-scoring that bootstraps PRISM's own judgment until it converges with Claude's.
4. **Cortex Loop**: layer duplication for deeper reasoning capacity, applied post-convergence. Cortex Loop is inspired by the [RYS (Repeat Your Self)](https://github.com/dnhkng/RYS) technique.
5. **Neuroscience-inspired training**: Spaced Repetition, Active Recall, TMR Dream Consolidation, Interleaved Training, and a Mixed Knowledge dataset to maximize retention and prevent forgetting.

This is **not** RAG. This is **not** external memory injection. The model itself changes.

---

## Requirements

- Ubuntu 24.04
- NVIDIA GPU: A100 40GB or RTX 4090 24GB
- CUDA 12.4+
- RunPod or similar cloud GPU instance

---

## RunPod Deployment

### 1. Spin up the instance

On RunPod:
- Template: **RunPod PyTorch** (Ubuntu 22/24, CUDA 12.x)
- GPU: A100 40GB (recommended) or RTX 4090
- Disk: 100 GB minimum (model is ~29 GB, adapters + data add up)
- Expose port: **8000**

### 2. SSH into the instance

```bash
ssh root@<your-runpod-ip> -p <port>
```

### 3. Clone the repo

```bash
cd /workspace
git clone <your-repo-url> prism_inside
cd prism_inside
```

### 4. Run setup

```bash
bash setup.sh
```

This installs all dependencies including PyTorch with CUDA 12.4.

### 5. Configure environment

```bash
cp .env.example .env
nano .env
```

Fill in:
```
HF_TOKEN=hf_your_token_here     # from huggingface.co/settings/tokens
ANTHROPIC_API_KEY=               # from console.anthropic.com (required for MIRROR + Dream Consolidation)
WANDB_API_KEY=                   # optional, leave empty to disable
```

All other defaults are production-ready.

### 6. Activate the virtual environment

```bash
source .venv/bin/activate
```

### 7. Download the model

```bash
python scripts/download_model.py
```

This downloads Qwen2.5-14B-Instruct (~29 GB) to `./model/weights/qwen2.5-14b-instruct`.
You need a HuggingFace account because the model is gated. Accept the license at huggingface.co/Qwen/Qwen2.5-14B-Instruct.

### 8. Start the server

```bash
python main.py
```

The server starts on `http://0.0.0.0:8000`.

To run in the background:
```bash
nohup python main.py > logs/server.log 2>&1 &
```

### 9. Verify it's running

```bash
curl http://localhost:8000/health
# {"status": "ok"}

curl http://localhost:8000/status
# full JSON status including GPU memory, episode counts, etc.
```

---

## Chat with it

### Using the client directly

```python
from prism_inside_client import PrismInsideClient

client = PrismInsideClient("http://localhost:8000", session_id="my_session")
result = client.chat("Hello! What can you do?")
print(result["response"])

# Rate the response (this trains the model over time)
client.feedback(result["episode_id"], was_useful=True, score=0.9)
```

### Interactive terminal chat

```bash
python prism_inside_client.py http://localhost:8000 my_session
```

Commands during chat:
- `/status`: show server health
- `/feedback 0.9 great response`: rate the last reply
- `/reset`: clear session memory
- `/train`: manually trigger training
- `/quit`: exit

---

## API Reference

### `POST /chat`
```json
{
  "session_id": "my_session",
  "message": "What do you know about me?",
  "system_prompt": null
}
```
Returns: `{ "response": "...", "episode_id": "uuid", "session_id": "..." }`

### `POST /feedback`
```json
{
  "episode_id": "uuid-from-chat",
  "was_useful": true,
  "score": 0.85,
  "reason": "accurate and concise"
}
```
Returns: `{ "episode_id": "...", "new_fitness": 0.85 }`

### `GET /status`
Returns full system health including GPU memory, training schedule, episode counts.

### `GET /mirror-status`
Returns MIRROR Protocol status: oracle call count, rolling average delta, convergence window progress, estimated episodes to convergence.

### `GET /cortex-loop-status`
Returns Cortex Loop status: enabled flag, scan completion, best layer-duplication config, seam targeting info.

### `POST /admin/trigger-training`
Manually triggers a LoRA + Titans training run.
Requires `TRAINING_MIN_EPISODES` high-fitness episodes to be logged.

### `POST /admin/reset-session`
```json
{ "session_id": "my_session" }
```
Clears the Titans memory state for a session (start fresh).

---

## How training works

1. Every `/chat` call logs an episode to SQLite. The **Contradiction Engine** runs first. If the new episode conflicts with recent high-fitness episodes, it is blocked, deferred, or merged before storage.
2. Every `/feedback` call updates that episode's fitness score (0.0 to 1.0).
3. At **3 AM daily** (configurable via `TRAINING_CRON` in `.env`):
   1. **Dream Consolidation** clusters old high-fitness episodes by semantic similarity, asks Claude to compress each cluster into a single semantic memory, validates existing provisional memories against recent episodes, and promotes or prunes them.
   2. **DatasetBuilder** collects high-fitness episodes + confirmed semantic memories into a shuffled training set using the 60/30/10 split (see Mixed Knowledge below).
   3. **LoRA training** runs on the combined dataset.
   4. **Titans adapter** trains on the same data.
   5. New adapters are **hot-swapped** into the live model. No restart needed.
   6. A full nightly report is written to `logs/nightly_run_{date}.json`.
4. Tomorrow the model is slightly more "you."

---

## Dream Consolidation

Dream Consolidation compresses clusters of related episodic memories into validated **semantic memories**, distilled truths about the user for future LoRA training.

**Stages:**
- **Cluster**: groups similar episodes (cosine similarity >= 0.75 on character trigrams)
- **Consolidate**: asks Claude to compress each cluster into one semantic memory
- **Safeguard**: rejects semantics with LLM confidence < 0.80 or where the cluster is too varied
- **Validate**: provisional semantics must be reinforced by 3+ matching recent episodes before promotion
- **Prune**: provisional semantics older than 14 days with zero validations are deleted

Semantic memories are never injected from the start. They must **earn confirmation** through real usage patterns.

```bash
# Manually trigger consolidation
curl -X POST http://localhost:8000/admin/run-consolidation

# View confirmed + provisional semantics
curl http://localhost:8000/semantics

# View last consolidation report
curl http://localhost:8000/admin/consolidation-report
```

**Requires:** `ANTHROPIC_API_KEY` in `.env`. Without it, consolidation is silently skipped and training proceeds on episodes only.

---

## Contradiction Engine

The Contradiction Engine runs before every episode is logged. It compares the new episode against the user's recent history (last 90 days, fitness >= 0.5) and calls Claude on any high-similarity pair.

**Resolutions:**
- `keep_a`: existing memory wins, new episode is blocked from storage
- `keep_b`: new episode wins, existing episode's fitness is set to 0.0 (excluded from training)
- `merge`: both are combined, existing is zeroed, merged content is stored
- `keep_both`: not a real contradiction, both are stored

Users never see contradiction resolution. The response is always returned normally.

```bash
# View contradiction log
curl http://localhost:8000/contradictions
```

**Requires:** `ANTHROPIC_API_KEY` in `.env`. Without it, all episodes are stored normally.

### Manual training trigger

```bash
# Via API
curl -X POST http://localhost:8000/admin/trigger-training

# Via script
python scripts/run_training.py
python scripts/run_training.py --dry-run  # check counts without training
```

---

## MIRROR Protocol

MIRROR is a self-calibration system that bootstraps PRISM's ability to judge response quality without constant human feedback.

**How it works:**

1. For every `/chat` + `/feedback` episode, the Oracle (Claude) also scores the response independently.
2. PRISM's score and the Oracle's score are compared. The **delta** is tracked over a rolling window (`MIRROR_CONVERGENCE_WINDOW`, default 50 episodes).
3. When the rolling average delta drops below `MIRROR_CONVERGENCE_THRESHOLD` (default 0.30), PRISM has **converged**. It scores its own episodes without Oracle calls.
4. Once `MIRROR_CONVERGED=true` is set in `.env`, Oracle API calls stop entirely and PRISM runs autonomously.

**Context file:** `./data/mirror_context.md` describes the user's identity and goals. The Oracle reads this to calibrate scoring.

```bash
# Check convergence progress
curl http://localhost:8000/mirror-status

# Run MIRROR bootstrap script (generates seed episodes for calibration)
python scripts/mirror_bootstrap.py

# Run MIRROR autopilot (continuous Oracle scoring loop)
python scripts/mirror_autopilot.py
```

**Requires:** `ANTHROPIC_API_KEY` (or `MIRROR_ANTHROPIC_API_KEY`) in `.env`.

---

## Cortex Loop

Cortex Loop increases PRISM's reasoning depth by duplicating transformer layers at a carefully chosen "seam," a point in the network where duplicating produces maximal benefit with minimal incoherence.

Cortex Loop is inspired by the [RYS (Repeat Your Self)](https://github.com/dnhkng/RYS) technique.

**How it works:**

1. `scripts/cortex_loop_scan.py` sweeps candidate seam positions and layer counts, scoring each configuration on reasoning benchmarks.
2. The best config is saved to `./config/rys_config.json` and `./config/rys_seam.json`.
3. `scripts/cortex_loop_apply.py` applies the duplication, producing `./model_rys/` with extra layers.
4. With `CORTEX_LOOP_ENABLED=true` and `CORTEX_LOOP_SCAN_COMPLETE=true`, the loader uses the Cortex Loop model instead of the base model.

Apply Cortex Loop **after** MIRROR convergence. The base model must first have stable judgment before its architecture is expanded.

```bash
# Step 1: scan for best seam (runs on GPU, takes ~1 hour)
python scripts/cortex_loop_scan.py

# Step 2: apply duplication
python scripts/cortex_loop_apply.py

# Step 3: enable in .env
# CORTEX_LOOP_ENABLED=true
# CORTEX_LOOP_SCAN_COMPLETE=true

# Check status
curl http://localhost:8000/cortex-loop-status
```

---

## Neuroscience Memory Upgrades

PRISM's training pipeline is modeled on how the human brain consolidates and retrieves memory.

### Spaced Repetition

Episodes are replayed at **Fibonacci intervals** (1, 1, 2, 3, 5, 8, 13, 21, 34, 55 training cycles after first learning). This mirrors the spacing effect. Memories reviewed at increasing intervals are retained far longer than massed repetition.

```
SPACED_REPLAY_ENABLED=true
SPACED_REPLAY_MAX_PER_BATCH=50
```

### Active Recall

Before each training run, the model is **tested** on a sample of previously trained episodes. Episodes where the model's recall score falls below `ACTIVE_RECALL_THRESHOLD` (default 0.40) are re-queued for training. Testing strengthens retention more than passive re-reading.

```
ACTIVE_RECALL_ENABLED=true
ACTIVE_RECALL_THRESHOLD=0.40
ACTIVE_RECALL_SAMPLE_SIZE=30
```

### TMR Dream Consolidation

Targeted Memory Reactivation: during each nightly training run, episodes with the highest delta between their original score and their Oracle/feedback score (`TMR_SALIENT_DELTA_THRESHOLD`, default 1.5) are **over-represented** in the training batch. High-emotion, high-salience memories get extra consolidation passes, exactly as in human sleep.

```
TMR_ENABLED=true
TMR_SALIENT_DELTA_THRESHOLD=1.5
TMR_REACTIVATION_RATIO=0.20
```

### Interleaved Training

Instead of training on all episodes of one topic before moving to the next, the DatasetBuilder **interleaves** topics in round-robin order. Interleaving causes short-term difficulty but produces better long-term retention and transfer. This is a well-established finding in human learning research.

```
INTERLEAVE_ENABLED=true
```

### Mixed Knowledge Dataset (60/30/10 split)

Every training batch is composed of three sources to prevent catastrophic forgetting:

| Proportion | Source |
|---|---|
| ~60% | High-fitness episodic memories (user interactions) |
| ~30% | General knowledge Q&A (`./data/general/`) |
| ~10% | Spaced-replay episodes (Fibonacci-scheduled reruns) |

The general knowledge layer (`MIXED_KNOWLEDGE_RATIO=0.30`) keeps PRISM from narrowing into a pure user-model and losing world knowledge. The spaced-replay layer (`SPACED_REPLAY_RATIO=0.10`) continuously reinforces the highest-value past learnings.

```
MIXED_KNOWLEDGE_ENABLED=true
MIXED_KNOWLEDGE_RATIO=0.30
SPACED_REPLAY_RATIO=0.10
MIXED_KNOWLEDGE_PATH=./data/general
```

---

## Benchmarking

```bash
# Before training
python scripts/benchmark.py run --tag before

# After training
python scripts/benchmark.py run --tag after

# Compare
python scripts/benchmark.py compare before after
```

Benchmark heatmaps are saved to `./logs/` as PNG files for visual comparison across runs.

---

## Architecture notes

### Why QLoRA?
The base model (29 GB) won't fit in VRAM for full fine-tuning even on an A100.
QLoRA loads it in 4-bit (NF4) and trains only small adapter matrices (~50 MB).
Result: full fine-tuning quality at ~1/4 the memory cost.

### Why Titans adapter?
Standard transformer attention is ephemeral and resets each call.
The Titans adapter maintains a persistent memory bank across turns within a session,
and its parameters evolve through training across sessions.
It's grafted at the embedding layer as a hook, so the base model is never touched.

### Why not RAG?
RAG is retrieval. You fetch relevant documents and inject them as context.
The model doesn't *know* anything.
PRISM makes the model *learn*. The weights shift toward the patterns
that mattered. The knowledge is structural, not retrieved.

---

## Project structure

```
prism_inside/
├── main.py                         # entry point
├── config.py                       # all config from .env
├── pyproject.toml                  # package metadata + dependencies
├── requirements.txt                # pinned dependencies
├── setup.sh                        # RunPod setup script
├── prism_inside_client.py          # Python client + interactive terminal chat
├── MIRROR.md                       # MIRROR Protocol deep-dive documentation
├── .env.example                    # environment template
├── .gitignore
│
├── model/
│   ├── loader.py                   # load model + adapters (base + Cortex Loop)
│   ├── inference.py                # inference engine
│   └── quantize.py                 # 4-bit NF4 config
│
├── memory_adapter/
│   ├── titans_adapter.py           # Titans cross-attention adapter
│   ├── memory_state.py             # per-session memory manager
│   ├── gating.py                   # MaG gating mechanism
│   └── train_adapter.py            # Titans training loop
│
├── training/
│   ├── lora_trainer.py             # QLoRA fine-tuning pipeline
│   ├── data_builder.py             # build training datasets (60/30/10 split)
│   ├── curriculum.py               # episode selection + interleaving
│   ├── scheduler.py                # APScheduler cron jobs
│   ├── dream_consolidation.py      # cluster + compress episodic memories
│   ├── contradiction_engine.py     # detect + resolve contradictions
│   └── active_recall.py            # test retention, re-queue forgotten episodes
│
├── data/
│   ├── experience_log.py           # SQLite episode logger
│   ├── episode_store.py            # re-export shim
│   ├── semantic_store.py           # confirmed + provisional semantic memories
│   ├── schemas.py                  # Pydantic models
│   ├── mirror_context.md           # MIRROR Oracle context file
│   └── general/
│       └── common_knowledge.jsonl  # general Q&A for mixed knowledge training
│
├── server/
│   ├── routes.py                   # FastAPI app + all route definitions
│   ├── chat.py                     # /chat logic
│   ├── feedback.py                 # /feedback logic
│   ├── status.py                   # /status logic
│   ├── mirror_hook.py              # MIRROR Oracle hook (auto-scores episodes)
│   └── mirror_oracle.py            # Oracle client (calls Claude)
│
├── scripts/
│   ├── download_model.py           # download Qwen2.5-14B-Instruct from HF
│   ├── run_training.py             # manual training trigger
│   ├── benchmark.py                # before/after benchmark + heatmap output
│   ├── mirror_bootstrap.py         # generate seed episodes for MIRROR calibration
│   ├── mirror_autopilot.py         # continuous MIRROR Oracle scoring loop
│   ├── cortex_loop_scan.py         # scan for optimal Cortex Loop seam position
│   └── cortex_loop_apply.py        # apply layer duplication, produce model_rys/
│
├── utils/
│   ├── text_similarity.py          # shared trigram cosine similarity
│   ├── anthropic_client.py         # shared Anthropic client singleton
│   └── mirror_context.py           # shared context file loader
│
├── tests/
│   ├── test_adapter.py
│   ├── test_contradiction_engine.py
│   ├── test_dream_consolidation.py
│   ├── test_feedback.py
│   ├── test_inference.py
│   └── test_training.py
│
└── logs/                           # nightly reports + benchmark heatmaps (gitignored)
```

---

## Running tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

Tests run on CPU. No GPU required for the test suite.
