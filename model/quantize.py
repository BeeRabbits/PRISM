"""4-bit quantization configuration for Qwen2.5-7B-Instruct."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import BitsAndBytesConfig

import config


def get_bnb_config() -> BitsAndBytesConfig:
    """
    Return a BitsAndBytesConfig for 4-bit NF4 quantization with double quant.

    This configuration:
    - Stores weights in 4-bit NF4 format (optimal for normally-distributed weights)
    - Uses double quantization to reduce memory further
    - Computes in bfloat16 for numerical stability on Ampere+ GPUs
    """
    if config.QUANTIZATION != "4bit":
        raise ValueError(
            f"Only '4bit' quantization is currently supported, got: {config.QUANTIZATION}"
        )

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
