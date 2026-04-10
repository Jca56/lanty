#!/usr/bin/env python3
"""
Merge a trained LoRA adapter into the base Qwen2.5-3B-Instruct model
and export the merged weights ready for GGUF conversion.

The actual GGUF conversion happens via llama.cpp's convert_hf_to_gguf.py
which is called from the Lambda deploy script (it needs llama.cpp built).

Usage:
    python scripts/merge_and_export.py [--adapter models/lanty-qwen-lora] [--output models/lanty-qwen-merged]
"""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_ADAPTER = PROJECT_ROOT / "models" / "lanty-qwen-lora"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "lanty-qwen-merged"


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Base model HuggingFace ID",
    )
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER), help="Path to LoRA adapter")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output dir for merged model")
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # CPU is fine for merging
        trust_remote_code=True,
    )

    print(f"Loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(base, args.adapter)

    print("Merging adapter into base weights...")
    merged = model.merge_and_unload()

    print(f"Saving merged model to {args.output}")
    Path(args.output).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output, safe_serialization=True)

    # Save tokenizer alongside (needed for inference)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(args.output)

    print("\nDone! Merged model saved.")
    print("\nNext: convert to GGUF for llama.cpp inference:")
    print(f"  cd /path/to/llama.cpp")
    print(f"  python convert_hf_to_gguf.py {args.output} --outfile lanty-qwen-f16.gguf")
    print(f"  ./build/bin/llama-quantize lanty-qwen-f16.gguf lanty-qwen-q4_k_m.gguf Q4_K_M")


if __name__ == "__main__":
    main()
