# Cortex Loop: layer duplication for enhanced reasoning depth.
# Inspired by the RYS technique: https://github.com/dnhkng/RYS
"""
Cortex Loop Apply: create an enhanced model with duplicated reasoning layers.

Reads the best (i, j) config from config/rys_config.json and creates
a permanent Cortex Loop enhanced model at ./model_rys/ with real layer copies
(not pointers) so it can be serialized and loaded independently.

Usage:
    python scripts/cortex_loop_apply.py                     # Use config from scan
    python scripts/cortex_loop_apply.py --i 10 --j 14       # Override config manually
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
import config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Apply Cortex Loop layer duplication to create enhanced model")
    parser.add_argument("--i", type=int, default=None, help="Start layer for duplication")
    parser.add_argument("--j", type=int, default=None, help="End layer for duplication")
    parser.add_argument("--model-path", type=str, default=None, help="Override source model path")
    parser.add_argument("--output-path", type=str, default=None, help="Override output model path")
    args = parser.parse_args()

    config_path = Path(config.CORTEX_LOOP_CONFIG_PATH)
    seam_path = Path(config.CORTEX_LOOP_SEAM_PATH)
    model_path = args.model_path or config.MODEL_LOCAL_PATH
    output_path = args.output_path or config.CORTEX_LOOP_MODEL_PATH

    # Load config
    if args.i is not None and args.j is not None:
        i, j = args.i, args.j
        print(f"Using manual config: i={i}, j={j}")
    elif config_path.exists():
        cfg = json.loads(config_path.read_text())
        i, j = cfg["i"], cfg["j"]
        print(f"Using scan config: i={i}, j={j} (delta={cfg['combined_delta']:+.4f})")
    else:
        print("Error: no config found. Run cortex_loop_scan.py first or provide --i and --j.")
        sys.exit(1)

    # Load model
    print(f"\nLoading base model from {model_path}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Load on CPU for manipulation
        trust_remote_code=True,
    )

    original_count = model.config.num_hidden_layers
    print(f"Original model: {original_count} layers")

    # Validate
    if i < 0 or j > original_count or i >= j or j - i < 2:
        print(f"Error: invalid (i={i}, j={j}) for {original_count}-layer model.")
        sys.exit(1)

    block_size = j - i
    new_count = original_count + block_size
    print(f"Duplicating layers {i}-{j-1} ({block_size} layers)")
    print(f"New model will have {new_count} layers\n")

    # Create real copies of the duplicated layers
    print("Creating deep copies of reasoning circuit layers...")
    original_layers = list(model.model.layers)
    duplicated_layers = [copy.deepcopy(layer) for layer in original_layers[i:j]]

    # Build new layer list: original[:j] + copies[i:j] + original[j:]
    new_layers = list(original_layers[:j]) + duplicated_layers + list(original_layers[j:])
    model.model.layers = nn.ModuleList(new_layers)
    model.config.num_hidden_layers = new_count

    # Update layer_types to match new layer count (required by newer transformers)
    if hasattr(model.config, "layer_types") and model.config.layer_types is not None:
        base_type = model.config.layer_types[0] if model.config.layer_types else "full_attention"
        model.config.layer_types = [base_type] * new_count

    print(f"Layer duplication complete: {original_count} → {new_count} layers")

    # Verify model works
    print("\nVerifying enhanced model generates coherent output...")
    model.eval()
    device = "cpu"
    test_prompt = "<|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\n"
    enc = tokenizer(test_prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            enc["input_ids"],
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output[0][enc["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    print(f"  Test response: {response[:100]}...")

    if len(response) < 5:
        print("  WARNING: very short response. Model may need verification.")
    else:
        print("  Verification passed.")

    # Save enhanced model
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving enhanced model to {output_dir}...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print("  Model saved.")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    original_params_per_layer = total_params / new_count  # approximate
    added_params = original_params_per_layer * block_size

    # Save seam info
    seam_info = {
        "i": i,
        "j": j,
        "seam_entry_layer": j,  # First layer of the duplicated block
        "seam_exit_layer": j + block_size - 1,  # Last layer of the duplicated block
        "original_layers": original_count,
        "total_layers_after_cortex_loop": new_count,
        "block_size": block_size,
        "note": "These are the layers MIRROR LoRA should target first",
        "created_at": datetime.utcnow().isoformat(),
    }
    seam_path.parent.mkdir(parents=True, exist_ok=True)
    seam_path.write_text(json.dumps(seam_info, indent=2))

    print(f"\n{'='*50}")
    print(f"  Cortex Loop Application Complete")
    print(f"{'='*50}")
    print(f"  Original layers  : {original_count}")
    print(f"  Duplicated block : layers {i}-{j-1} ({block_size} layers)")
    print(f"  New total layers : {new_count}")
    print(f"  Total parameters : {total_params:,}")
    print(f"  ~Added params    : {added_params:,.0f}")
    print(f"  Seam entry layer : {j}")
    print(f"  Seam exit layer  : {j + block_size - 1}")
    print(f"  Model saved to   : {output_dir}")
    print(f"  Seam info saved  : {seam_path}")
    print()
    print("  Next steps:")
    print("  1. Add to .env: CORTEX_LOOP_ENABLED=true")
    print("  2. Add to .env: CORTEX_LOOP_SCAN_COMPLETE=true")
    print("  3. Restart PRISM server")
    print(f"  4. The model will load from {output_dir}")
    print()
    print("  NOTE: Existing LoRA adapters are incompatible with the Cortex Loop model")
    print("  (layer indices changed). Fresh LoRA training will start automatically.")


if __name__ == "__main__":
    main()
