#!/usr/bin/env python3
"""
LoRA fine-tune Qwen2.5-7B-Instruct on Lanty dialogue data.

Uses HuggingFace transformers + peft directly (NOT trl's SFTTrainer, which
has shown NaN gradient issues with newer torch/peft versions). We build the
dataset manually with the chat template and use the standard Trainer.

Full bf16 LoRA — no QLoRA. 4-bit base (bitsandbytes) has shown NaN issues
with newer torch/bnb versions, and Lambda H100s have plenty of VRAM anyway.

Designed to run on a Lambda Cloud GPU instance.

Usage:
    python scripts/train_lanty.py
    python scripts/train_lanty.py --epochs 3 --lr 1e-4 --batch-size 4
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "lanty_chat.jsonl"
EVAL_PATH = PROJECT_ROOT / "data" / "lanty_chat_eval.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "models" / "lanty-qwen-lora"


class NaNGuard(TrainerCallback):
    """Abort training if any LoRA gradient or parameter goes NaN.

    Runs at the end of every logging step (cheap — only iterates trainable
    params, which is 1-2% of total for LoRA). Catches NaN early so we don't
    burn an H100 hour on a doomed run.
    """

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % args.logging_steps != 0:
            return
        model = kwargs.get("model")
        if model is None:
            return
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is not None and torch.isnan(p.grad).any():
                print(f"\n!!! NaN gradient detected at step {state.global_step} in {name}. Aborting.")
                control.should_training_stop = True
                return
            if torch.isnan(p).any():
                print(f"\n!!! NaN parameter detected at step {state.global_step} in {name}. Aborting.")
                control.should_training_stop = True
                return


def load_examples(path: Path) -> list[dict]:
    examples = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def build_dataset(examples: list[dict], tokenizer, max_length: int) -> Dataset:
    """Apply chat template and tokenize each example.

    We build input_ids and labels manually so we know exactly what the model sees.
    Labels match input_ids — loss is computed on every token (including system/user).
    Pad token positions are masked to -100.
    """
    pad_id = tokenizer.pad_token_id

    def process(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        # Labels mirror input_ids; mask pad tokens so they don't contribute to loss
        labels = [tok if tok != pad_id else -100 for tok in input_ids]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    raw_dataset = Dataset.from_list(examples)
    return raw_dataset.map(process, remove_columns=raw_dataset.column_names)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5 on Lanty dialogues")
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model ID for the base model",
    )
    parser.add_argument("--data", default=str(DATA_PATH), help="Path to training JSONL")
    parser.add_argument("--eval-data", default=str(EVAL_PATH), help="Path to eval JSONL (optional)")
    parser.add_argument("--output", default=str(OUTPUT_DIR), help="Output directory for LoRA adapter")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs (default: 5)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size (default: 2 for 7B model)")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps (default: 8)")
    parser.add_argument("--max-seq-len", type=int, default=1536, help="Max sequence length (default: 1536)")
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA rank (default: 64)")
    parser.add_argument("--lora-alpha", type=int, default=128, help="LoRA alpha (default: 128)")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout (default: 0.05)")
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("Lanty LoRA Fine-Tuning (manual Trainer)")
    print("=" * 60)
    print(f"Base model:    {args.base_model}")
    print(f"Data:          {args.data}")
    print(f"Output:        {args.output}")
    print(f"Epochs:        {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size:    {args.batch_size} (x{args.grad_accum} grad accum)")
    print(f"Max seq len:   {args.max_seq_len}")
    print(f"LoRA:          r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print()

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("FATAL: No CUDA GPU detected. This script requires a GPU.")
        return
    print()

    # Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Pad token id: {tokenizer.pad_token_id}")

    # Datasets
    print("\nLoading dataset...")
    examples = load_examples(Path(args.data))
    print(f"  Train: {len(examples)} examples")
    dataset = build_dataset(examples, tokenizer, args.max_seq_len)
    print(f"  Tokenized to {len(dataset)} sequences of {args.max_seq_len} tokens")

    eval_dataset = None
    eval_path = Path(args.eval_data)
    if eval_path.exists():
        eval_examples = load_examples(eval_path)
        print(f"  Eval:  {len(eval_examples)} examples")
        eval_dataset = build_dataset(eval_examples, tokenizer, args.max_seq_len)
    else:
        print(f"  Eval:  (no eval set at {eval_path}, skipping eval)")

    # Sanity check the first example
    first = dataset[0]
    label_count = sum(1 for l in first["labels"] if l != -100)
    print(f"  First example: {len(first['input_ids'])} tokens, {label_count} non-pad labels")
    print()

    # Model in bf16 (no quantization)
    print("Loading base model in bf16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.config.use_cache = False  # save memory during training

    # LoRA on attention + MLP projections
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} ({100 * trainable / total:.2f}% of {total:,})")
    print()

    # Standard HF TrainingArguments — no SFTTrainer, no surprises
    training_kwargs = dict(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=True,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
        gradient_checkpointing=False,
    )
    if eval_dataset is not None:
        training_kwargs.update(
            eval_strategy="steps",
            eval_steps=args.save_steps,
            per_device_eval_batch_size=args.batch_size,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
    training_args = TrainingArguments(**training_kwargs)

    # Plain language modeling collator (we already padded + set labels manually,
    # so this just stacks the tensors)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[NaNGuard()],
    )

    print("=" * 60)
    print("Starting training...")
    print("=" * 60)
    trainer.train()

    print("\nSaving LoRA adapter...")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    # Sanity check: are saved weights NaN?
    from safetensors.torch import load_file
    adapter_path = Path(args.output) / "adapter_model.safetensors"
    if adapter_path.exists():
        weights = load_file(str(adapter_path))
        nan_count = sum(1 for t in weights.values() if torch.isnan(t).any())
        print(f"  Tensors with NaN: {nan_count}/{len(weights)}")
        if nan_count > 0:
            print("  WARNING: Some tensors contain NaN. Training likely failed.")
        else:
            print("  All weights are finite. Ready to merge + convert to GGUF.")

    print(f"\nDone! LoRA adapter saved to {args.output}")
    print(f"\nNext steps:")
    print(f"  1. Merge LoRA into base model:  python scripts/merge_and_export.py")
    print(f"  2. Convert to GGUF for llama.cpp inference")


if __name__ == "__main__":
    main()
