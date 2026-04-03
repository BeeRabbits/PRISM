"""
ModelLoader: loads Qwen2.5-7B-Instruct with optional LoRA and Titans adapters.

The loader handles:
  1. Downloading the base model if not present locally.
  2. Loading with 4-bit quantization.
  3. Attaching an existing LoRA adapter (PEFT) if available.
  4. Attaching the Titans memory adapter if available.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

import config
from model.quantize import get_bnb_config

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Loads and holds the base model, tokenizer, and optional adapters.

    Attributes:
        model: The loaded (possibly adapter-wrapped) causal LM.
        tokenizer: The associated tokenizer.
    """

    def __init__(self) -> None:
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self._lora_loaded: bool = False
        self._titans_loaded: bool = False
        self._adapter_version: str = "none"
        self._cortex_loop_active: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Full load sequence:
          1. Download model if needed.
          2. Load with 4-bit quantization.
          3. Attach LoRA adapter if present.
          4. Attach Titans adapter if present.
        """
        self._ensure_downloaded()
        self._load_base_model()
        self._attach_lora_adapter()
        self._attach_titans_adapter()
        logger.info("ModelLoader.load() complete. Model is ready.")

    @property
    def is_ready(self) -> bool:
        """Return True if model and tokenizer are both loaded."""
        return self.model is not None and self.tokenizer is not None

    @property
    def adapter_version(self) -> str:
        return self._adapter_version

    @property
    def lora_loaded(self) -> bool:
        return self._lora_loaded

    @property
    def titans_loaded(self) -> bool:
        return self._titans_loaded

    @property
    def cortex_loop_active(self) -> bool:
        return self._cortex_loop_active

    def hot_swap_lora(self) -> None:
        """
        Reload the LoRA adapter from disk without restarting the server.
        Replaces the current adapter in-place on the running model.
        """
        if not self._lora_loaded:
            logger.warning("hot_swap_lora called but no LoRA adapter was previously loaded.")
        logger.info("Hot-swapping LoRA adapter from %s", config.ADAPTER_PATH)
        self._attach_lora_adapter()
        logger.info("LoRA adapter hot-swap complete.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_downloaded(self) -> None:
        """Download the model from HuggingFace if not already present."""
        local = Path(config.MODEL_LOCAL_PATH)
        if local.exists() and any(local.glob("*.safetensors")):
            logger.info("Model weights found at %s. Skipping download.", local)
            return

        logger.info("Model not found locally. Downloading %s...", config.MODEL_ID)
        from scripts.download_model import download_model
        download_model()

    def _load_base_model(self) -> None:
        """Load tokenizer and 4-bit quantized base model, then apply Cortex Loop at runtime."""
        if config.CORTEX_LOOP_ENABLED and not config.CORTEX_LOOP_SCAN_COMPLETE:
            logger.warning(
                "Cortex Loop enabled but scan not complete. "
                "Run: python scripts/cortex_loop_scan.py"
            )
        # Always load from the original base model (Cortex Loop applied at runtime)
        local = str(Path(config.MODEL_LOCAL_PATH).resolve())
        logger.info("Loading tokenizer from %s", local)

        self.tokenizer = AutoTokenizer.from_pretrained(
            local,
            trust_remote_code=True,
        )
        # Ensure pad token is set (Qwen uses eos as pad by default)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Loading base model with 4-bit quantization...")
        bnb_config = get_bnb_config()

        self.model = AutoModelForCausalLM.from_pretrained(
            local,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self.model.config.use_cache = False  # required for gradient checkpointing
        device_map = getattr(self.model, "hf_device_map", "auto")
        logger.info("Base model loaded. Device map: %s", device_map)

        # Cortex Loop: apply layer duplication at runtime via pointer manipulation.
        # This works with quantized models because we reuse the already-loaded
        # layer objects. Zero extra VRAM, just an extra forward pass through
        # the reasoning circuit.
        if config.CORTEX_LOOP_ENABLED and config.CORTEX_LOOP_SCAN_COMPLETE:
            self._apply_cortex_loop_runtime()

    def _apply_cortex_loop_runtime(self) -> None:
        """Apply Cortex Loop layer duplication at runtime using pointer manipulation."""
        import json as _json
        from torch.nn import ModuleList

        cl_config_path = Path(config.CORTEX_LOOP_CONFIG_PATH)
        if not cl_config_path.exists():
            logger.warning("Cortex Loop config not found at %s. Skipping.", cl_config_path)
            return

        cfg = _json.loads(cl_config_path.read_text())
        # Support both flat format {"i": X, "j": Y} and nested {"reasoning": {"i": X, "j": Y}}
        if "reasoning" in cfg:
            rcfg = cfg["reasoning"]
        else:
            rcfg = cfg
        i, j = rcfg["i"], rcfg["j"]
        original_layers = list(self.model.model.layers)
        original_count = len(original_layers)

        if i < 0 or j > original_count or i >= j:
            logger.error("Cortex Loop config (i=%d, j=%d) invalid for %d-layer model.", i, j, original_count)
            return

        # Duplicate layers[i:j] via pointers (zero VRAM, same layer objects)
        new_layers = list(original_layers[:j]) + list(original_layers[i:j]) + list(original_layers[j:])
        self.model.model.layers = ModuleList(new_layers)
        self.model.config.num_hidden_layers = len(new_layers)
        self._cortex_loop_active = True

        logger.info(
            "Cortex Loop applied at runtime: layers %d-%d duplicated (%d → %d layers, zero extra VRAM)",
            i, j - 1, original_count, len(new_layers),
        )

    def _attach_lora_adapter(self) -> None:
        """Attach a PEFT LoRA adapter if ADAPTER_PATH exists and has adapter files."""
        adapter_path = Path(config.ADAPTER_PATH)
        if not adapter_path.exists():
            logger.info("No LoRA adapter found at %s. Skipping.", adapter_path)
            return

        # Check for adapter_config.json as a signal that a valid adapter exists
        if not (adapter_path / "adapter_config.json").exists():
            logger.info(
                "ADAPTER_PATH exists but no adapter_config.json. Skipping LoRA attach."
            )
            return

        logger.info("Attaching LoRA adapter from %s", adapter_path)

        # If model is already wrapped in PeftModel, properly unload first
        if isinstance(self.model, PeftModel):
            self.model = self.model.unload()

        # Clean leftover peft_config from training (get_peft_model modifies
        # the base model in-place). Without this, from_pretrained stacks
        # a second adapter, producing Chinese/garbage output.
        if hasattr(self.model, "peft_config"):
            delattr(self.model, "peft_config")

        self.model = PeftModel.from_pretrained(
            self.model,
            str(adapter_path),
            is_trainable=True,  # keep adapter separate for further training
            torch_dtype=torch.bfloat16,
        )
        self._lora_loaded = True
        self._adapter_version = str(adapter_path.stat().st_mtime)
        logger.info("LoRA adapter attached. Version timestamp: %s", self._adapter_version)

    def _attach_titans_adapter(self) -> None:
        """Attach Titans memory adapter if trained adapter exists."""
        titans_path = Path(config.TITANS_ADAPTER_PATH)
        if not titans_path.exists():
            logger.info("No Titans adapter found at %s. Skipping.", titans_path)
            return

        checkpoint = titans_path / "titans_adapter.pt"
        if not checkpoint.exists():
            logger.info("Titans adapter path exists but no titans_adapter.pt. Skipping.")
            return

        logger.info("Attaching Titans memory adapter from %s", checkpoint)
        from memory_adapter.titans_adapter import TitansMemoryAdapter

        titans = TitansMemoryAdapter(
            model_hidden_size=self.model.config.hidden_size,
            memory_size=config.TITANS_MEMORY_SIZE,
            memory_dim=config.TITANS_MEMORY_DIM,
        )
        state = torch.load(str(checkpoint), map_location="cpu")
        titans.load_state_dict(state)
        titans.eval()

        # Match device and dtype to the model
        param = next(self.model.parameters())
        titans = titans.to(device=param.device, dtype=param.dtype)

        # Register as a named submodule so it's part of the model graph
        self.model.titans_adapter = titans
        self._titans_loaded = True
        logger.info("Titans adapter loaded.")
