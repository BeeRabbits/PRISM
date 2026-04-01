"""Central configuration loaded from .env."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (wherever this file lives)
_ROOT = Path(__file__).parent
load_dotenv(_ROOT / ".env")


def _get(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _getfloat(key: str, default: float = 0.0) -> float:
    return float(os.getenv(key, str(default)))


def _getint(key: str, default: int = 0) -> int:
    return int(os.getenv(key, str(default)))


# Model
HF_TOKEN: str = _get("HF_TOKEN")
MODEL_ID: str = _get("MODEL_ID", "Qwen/Qwen2.5-14B-Instruct")
MODEL_LOCAL_PATH: str = _get("MODEL_LOCAL_PATH", "./model/weights/qwen2.5-14b-instruct")

# Adapters
ADAPTER_PATH: str = _get("ADAPTER_PATH", "./adapters/current")
TITANS_ADAPTER_PATH: str = _get("TITANS_ADAPTER_PATH", "./adapters/titans")

# Database
EXPERIENCE_DB_URL: str = _get(
    "EXPERIENCE_DB_URL", "sqlite+aiosqlite:///./data/experience.db"
)

# Weights & Biases
WANDB_API_KEY: str = _get("WANDB_API_KEY", "")
WANDB_PROJECT: str = _get("WANDB_PROJECT", "prism")

# Server
SERVER_PORT: int = _getint("SERVER_PORT", 8000)

# Hardware
DEVICE: str = _get("DEVICE", "cuda")
QUANTIZATION: str = _get("QUANTIZATION", "4bit")

# LoRA
LORA_RANK: int = _getint("LORA_RANK", 16)
LORA_ALPHA: int = _getint("LORA_ALPHA", 32)
LORA_DROPOUT: float = _getfloat("LORA_DROPOUT", 0.05)
LORA_TARGET_MODULES: list[str] = _get(
    "LORA_TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj"
).split(",")

# Titans
TITANS_MEMORY_DIM: int = _getint("TITANS_MEMORY_DIM", 512)
TITANS_MEMORY_LAYERS: int = _getint("TITANS_MEMORY_LAYERS", 0)
TITANS_GATING_METHOD: str = _get("TITANS_GATING_METHOD", "MaG")
TITANS_MEMORY_SIZE: int = _getint("TITANS_MEMORY_SIZE", 512)
TITANS_MOMENTUM: float = _getfloat("TITANS_MOMENTUM", 0.9)

# Training thresholds
TRAINING_MIN_FITNESS: float = _getfloat("TRAINING_MIN_FITNESS", 0.6)
TRAINING_MIN_EPISODES: int = _getint("TRAINING_MIN_EPISODES", 50)
TRAINING_CRON: str = _get("TRAINING_CRON", "0 3 * * *")

# Training hyperparameters
MAX_SEQ_LENGTH: int = _getint("MAX_SEQ_LENGTH", 4096)
BATCH_SIZE: int = _getint("BATCH_SIZE", 2)
GRAD_ACCUM_STEPS: int = _getint("GRAD_ACCUM_STEPS", 4)
LEARNING_RATE: float = _getfloat("LEARNING_RATE", 2e-4)
WARMUP_RATIO: float = _getfloat("WARMUP_RATIO", 0.03)
NUM_EPOCHS: int = _getint("NUM_EPOCHS", 1)

# Anthropic API (used by Dream Consolidation + Contradiction Engine)
ANTHROPIC_API_KEY: str = _get("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL: str = _get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# Dream Consolidation
CONSOLIDATION_MIN_FITNESS: float = _getfloat("CONSOLIDATION_MIN_FITNESS", 0.6)
CONSOLIDATION_MIN_CLUSTER_SIZE: int = _getint("CONSOLIDATION_MIN_CLUSTER_SIZE", 3)
CONSOLIDATION_SIMILARITY_THRESHOLD: float = _getfloat("CONSOLIDATION_SIMILARITY_THRESHOLD", 0.75)
CONSOLIDATION_MIN_CONFIDENCE: float = _getfloat("CONSOLIDATION_MIN_CONFIDENCE", 0.80)

# Contradiction Engine
CONTRADICTION_SIMILARITY_THRESHOLD: float = _getfloat("CONTRADICTION_SIMILARITY_THRESHOLD", 0.80)
CONTRADICTION_MIN_CONFIDENCE: float = _getfloat("CONTRADICTION_MIN_CONFIDENCE", 0.75)

# Semantic memory validation
SEMANTIC_VALIDATION_THRESHOLD: int = _getint("SEMANTIC_VALIDATION_THRESHOLD", 3)
SEMANTIC_PRUNE_DAYS: int = _getint("SEMANTIC_PRUNE_DAYS", 14)

# MIRROR Protocol
MIRROR_ENABLED: bool = _get("MIRROR_ENABLED", "true").lower() in ("true", "1", "yes")
MIRROR_ORACLE_MODEL: str = _get("MIRROR_ORACLE_MODEL", "claude-sonnet-4-6")
MIRROR_ANTHROPIC_API_KEY: str = _get("MIRROR_ANTHROPIC_API_KEY") or ANTHROPIC_API_KEY
MIRROR_CONTEXT_FILE: str = _get("MIRROR_CONTEXT_FILE", "./data/mirror_context.md")
MIRROR_CONVERGENCE_THRESHOLD: float = _getfloat("MIRROR_CONVERGENCE_THRESHOLD", 0.30)
MIRROR_CONVERGENCE_WINDOW: int = _getint("MIRROR_CONVERGENCE_WINDOW", 50)
MIRROR_CONVERGED: bool = _get("MIRROR_CONVERGED", "false").lower() in ("true", "1", "yes")

# Cortex Loop (layer duplication for reasoning depth)
CORTEX_LOOP_ENABLED: bool = _get("CORTEX_LOOP_ENABLED", "false").lower() in ("true", "1", "yes")
CORTEX_LOOP_MODEL_PATH: str = _get("CORTEX_LOOP_MODEL_PATH", "./model_rys")
CORTEX_LOOP_CONFIG_PATH: str = _get("CORTEX_LOOP_CONFIG_PATH", "./config/rys_config.json")
CORTEX_LOOP_SEAM_PATH: str = _get("CORTEX_LOOP_SEAM_PATH", "./config/rys_seam.json")
CORTEX_LOOP_SCAN_COMPLETE: bool = _get("CORTEX_LOOP_SCAN_COMPLETE", "false").lower() in ("true", "1", "yes")

# Spaced Repetition: Fibonacci-interval replay of trained episodes
SPACED_REPLAY_ENABLED: bool = _get("SPACED_REPLAY_ENABLED", "true").lower() in ("true", "1", "yes")
SPACED_REPLAY_MAX_PER_BATCH: int = _getint("SPACED_REPLAY_MAX_PER_BATCH", 50)
SPACED_REPLAY_FIBONACCI: list[int] = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]

# Active Recall: test model retention and re-queue forgotten episodes
ACTIVE_RECALL_ENABLED: bool = _get("ACTIVE_RECALL_ENABLED", "true").lower() in ("true", "1", "yes")
ACTIVE_RECALL_THRESHOLD: float = _getfloat("ACTIVE_RECALL_THRESHOLD", 0.40)
ACTIVE_RECALL_SAMPLE_SIZE: int = _getint("ACTIVE_RECALL_SAMPLE_SIZE", 30)

# TMR Dream Consolidation: prioritize salient (high-delta) memories
TMR_ENABLED: bool = _get("TMR_ENABLED", "true").lower() in ("true", "1", "yes")
TMR_SALIENT_DELTA_THRESHOLD: float = _getfloat("TMR_SALIENT_DELTA_THRESHOLD", 1.5)
TMR_REACTIVATION_RATIO: float = _getfloat("TMR_REACTIVATION_RATIO", 0.20)

# Interleaved Training: round-robin topic ordering to prevent distribution collapse
INTERLEAVE_ENABLED: bool = _get("INTERLEAVE_ENABLED", "true").lower() in ("true", "1", "yes")

# Mixed Knowledge: blend general knowledge to prevent catastrophic forgetting
MIXED_KNOWLEDGE_ENABLED: bool = _get("MIXED_KNOWLEDGE_ENABLED", "true").lower() in ("true", "1", "yes")
MIXED_KNOWLEDGE_RATIO: float = _getfloat("MIXED_KNOWLEDGE_RATIO", 0.30)
SPACED_REPLAY_RATIO: float = _getfloat("SPACED_REPLAY_RATIO", 0.10)
MIXED_KNOWLEDGE_PATH: str = _get("MIXED_KNOWLEDGE_PATH", "./data/general")

# Frustration Detection (regex-based, zero API calls)
FRUSTRATION_DETECTION_ENABLED: bool = _get("FRUSTRATION_DETECTION_ENABLED", "true").lower() in ("true", "1", "yes")

# Idle-time Dream Consolidation
IDLE_CONSOLIDATION_ENABLED: bool = _get("IDLE_CONSOLIDATION_ENABLED", "true").lower() in ("true", "1", "yes")
IDLE_CONSOLIDATION_MINUTES: int = _getint("IDLE_CONSOLIDATION_MINUTES", 30)
IDLE_CONSOLIDATION_COOLDOWN_MINUTES: int = _getint("IDLE_CONSOLIDATION_COOLDOWN_MINUTES", 120)

# Memory Validation and Injection
MEMORY_INJECTION_ENABLED: bool = _get("MEMORY_INJECTION_ENABLED", "true").lower() in ("true", "1", "yes")
MEMORY_VALIDATION_MIN_SOURCE_FITNESS: float = _getfloat("MEMORY_VALIDATION_MIN_SOURCE_FITNESS", 0.5)
MEMORY_VALIDATION_MAX_STALENESS_DAYS: int = _getint("MEMORY_VALIDATION_MAX_STALENESS_DAYS", 30)
MEMORY_VALIDATION_MIN_CONFIDENCE: float = _getfloat("MEMORY_VALIDATION_MIN_CONFIDENCE", 0.50)

# Instincts System (learned behavioral rules)
INSTINCTS_ENABLED: bool = _get("INSTINCTS_ENABLED", "true").lower() in ("true", "1", "yes")
INSTINCTS_MAX_ACTIVE: int = _getint("INSTINCTS_MAX_ACTIVE", 20)
INSTINCTS_MIN_CONFIDENCE: float = _getfloat("INSTINCTS_MIN_CONFIDENCE", 0.60)

# Context Compaction (session history summarization)
COMPACTION_ENABLED: bool = _get("COMPACTION_ENABLED", "true").lower() in ("true", "1", "yes")
COMPACTION_TOKEN_THRESHOLD: int = _getint("COMPACTION_TOKEN_THRESHOLD", 3000)
COMPACTION_KEEP_RECENT: int = _getint("COMPACTION_KEEP_RECENT", 6)

# User identity (used by autopilot and display, no personal info in code)
USER_NAME: str = _get("USER_NAME", "User")
