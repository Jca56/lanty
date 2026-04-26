#!/usr/bin/env python3
"""
Per-mode voice quality judge for Lanty's training data.

Samples N synthetic dialogues per mode, asks Claude to score each on
voice match, length variety, lore accuracy (lockedin only), and mode
adherence. Writes a JSON report with per-mode aggregates and the
lowest-scoring 3 samples per mode for inspection.

Uses the SAME prompt cache as lanty_gen/api.py — the heavy ~100KB
context (personality + gold samples) gets read at ~10% cost after the
first call. A 50/mode run usually lands at $1-3.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/eval_voice.py [--per-mode 50] [--max-cost 10.00]
    python scripts/eval_voice.py --yes      # skip cost gate
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

from lanty_gen.prompts import (
    DATA_DIR,
    build_system_prompt,
    load_gold_samples,
    load_lore_context,
    load_personality,
)

TRAINING_DIR = DATA_DIR / "training"
EVAL_DIR = DATA_DIR / "eval"
MODEL = "claude-sonnet-4-6"

# Rough Claude pricing (sonnet-4-6, USD per million tokens, Apr 2026)
PRICE_INPUT = 3.00
PRICE_CACHE_WRITE = 3.75
PRICE_CACHE_READ = 0.30
PRICE_OUTPUT = 15.00

DIALOGUE_PATTERN = re.compile(
    r"<player>(.*?)</player>\s*<lanty>(.*?)</lanty>",
    re.DOTALL,
)


def collect_samples(per_mode: int, seed: int) -> dict[str, list[dict]]:
    """Sample up to `per_mode` dialogues from each mode's batch_*.txt files."""
    rng = random.Random(seed)
    by_mode: dict[str, list[dict]] = defaultdict(list)

    for f in sorted(TRAINING_DIR.glob("batch_*.txt")):
        m = re.match(r"^batch_(\d{4})_(.+)\.txt$", f.name)
        if not m:
            continue
        mode = m.group(2)
        text = f.read_text()
        for player_msg, lanty_msg in DIALOGUE_PATTERN.findall(text):
            by_mode[mode].append({
                "source": f.name,
                "player": player_msg.strip(),
                "lanty": lanty_msg.strip(),
            })

    sampled: dict[str, list[dict]] = {}
    for mode, items in by_mode.items():
        rng.shuffle(items)
        sampled[mode] = items[:per_mode]
    return sampled


def build_judge_prompt(mode: str, sample: dict) -> str:
    is_lockedin = mode == "lockedin"
    criteria = [
        '"voice_match": 1-5 — does Lanty sound like Lanty (speech patterns, energy, warmth)?',
        '"length_appropriate": 1-5 — is the response length appropriate for the player line?',
        '"mode_adherence": 1-5 — does it stay in the requested mode (' + mode + ')?',
    ]
    if is_lockedin:
        criteria.append('"lore_accuracy": 1-5 — is any Flamebound lore claim factually correct per the world spec?')

    return f"""Score the following Lanty dialogue exchange. Mode: {mode}.

<player>{sample['player']}</player>
<lanty>{sample['lanty']}</lanty>

Return ONLY a JSON object with these fields, nothing else:
{{
  {chr(10).join("  " + c + "," for c in criteria)}
  "notes": "one short sentence on the weakest aspect"
}}"""


def estimate_cost(n_calls: int, system_chars: int) -> float:
    """Rough cost estimate for n_calls with prompt caching on the system block."""
    system_tokens = system_chars / 4  # rough
    user_tokens = 200  # judge prompt + sample
    output_tokens = 100  # JSON response

    cache_write_cost = (system_tokens * PRICE_CACHE_WRITE) / 1e6
    cache_read_cost = (system_tokens * PRICE_CACHE_READ * (n_calls - 1)) / 1e6
    user_cost = (user_tokens * n_calls * PRICE_INPUT) / 1e6
    output_cost = (output_tokens * n_calls * PRICE_OUTPUT) / 1e6
    return cache_write_cost + cache_read_cost + user_cost + output_cost


def score_one(client, system, mode, sample) -> tuple[str, dict, dict]:
    """Returns (mode, scores_dict, usage_dict). On parse failure, scores={}."""
    prompt = build_judge_prompt(mode, sample)
    response = client.messages.create(
        model=MODEL,
        max_tokens=400,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    usage = {
        "input": response.usage.input_tokens,
        "cache_read": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        "cache_write": getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
        "output": response.usage.output_tokens,
    }
    try:
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
        scores = json.loads(text)
    except json.JSONDecodeError:
        scores = {"_parse_error": text[:200]}
    return mode, scores, usage


def main():
    parser = argparse.ArgumentParser(description="Voice quality judge for Lanty training data")
    parser.add_argument("--per-mode", type=int, default=50, help="Samples per mode (default: 50)")
    parser.add_argument("--workers", type=int, default=3, help="Parallel API workers (default: 3)")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (default: 42)")
    parser.add_argument("--max-cost", type=float, default=10.00, help="Abort if estimated cost exceeds this (USD, default: 10)")
    parser.add_argument("--yes", action="store_true", help="Skip the cost confirmation prompt")
    parser.add_argument("--output-dir", type=str, default=str(EVAL_DIR), help="Output directory")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    print("Loading lore + personality + gold samples for the judge prompt...")
    system = build_system_prompt(load_personality(), load_gold_samples(), load_lore_context())
    print(f"  System prompt: {len(system):,} chars")

    print("\nSampling dialogues...")
    samples = collect_samples(args.per_mode, args.seed)
    total = sum(len(v) for v in samples.values())
    for mode in sorted(samples):
        print(f"  {mode:<14} {len(samples[mode])}")
    print(f"  {'total':<14} {total}")

    estimated = estimate_cost(total, len(system))
    print(f"\nEstimated cost (with prompt caching): ${estimated:.2f}")
    if estimated > args.max_cost:
        print(f"  ABORT: estimate ${estimated:.2f} exceeds --max-cost ${args.max_cost:.2f}")
        sys.exit(1)
    if not args.yes:
        reply = input(f"Proceed with ${estimated:.2f} estimate? [y/N] ").strip().lower()
        if reply != "y":
            print("Aborted.")
            return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic(api_key=api_key)

    print(f"\nScoring {total} samples with {args.workers} workers...")
    results: dict[str, list] = defaultdict(list)
    totals = {"input": 0, "cache_read": 0, "cache_write": 0, "output": 0}
    start = time.time()
    completed = 0

    jobs = [(mode, sample) for mode, items in samples.items() for sample in items]

    def worker(job):
        mode, sample = job
        try:
            return score_one(client, system, mode, sample) + (sample,)
        except Exception as e:
            return (mode, {"_error": str(e)}, {}, sample)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(worker, j) for j in jobs]
        for fut in as_completed(futures):
            mode, scores, usage, sample = fut.result()
            results[mode].append({**sample, **scores})
            for k, v in usage.items():
                totals[k] = totals.get(k, 0) + v
            completed += 1
            if completed % 10 == 0 or completed == len(jobs):
                elapsed = time.time() - start
                print(f"  [{completed}/{len(jobs)}] {elapsed:.0f}s elapsed")

    # Aggregate
    print("\n=== Per-mode summary ===")
    summary: dict[str, dict] = {}
    for mode, items in sorted(results.items()):
        valid = [it for it in items if "_error" not in it and "_parse_error" not in it]
        if not valid:
            print(f"  {mode}: NO VALID SCORES")
            continue
        score_keys = ["voice_match", "length_appropriate", "mode_adherence"]
        if mode == "lockedin":
            score_keys.append("lore_accuracy")
        means: dict[str, float] = {}
        for k in score_keys:
            vals = [it.get(k) for it in valid if isinstance(it.get(k), (int, float))]
            means[k] = sum(vals) / len(vals) if vals else 0.0
        summary[mode] = {"n": len(valid), "means": means}

        items_sorted = sorted(
            valid,
            key=lambda it: sum(it.get(k, 5) for k in score_keys if isinstance(it.get(k), (int, float))),
        )
        worst = items_sorted[:3]
        score_str = "  ".join(f"{k}={v:.2f}" for k, v in means.items())
        print(f"  {mode:<14} n={len(valid):>3}  {score_str}")
        for w in worst:
            print(f"    weak: <{w['source']}> {w.get('notes', '?')[:80]}")

    actual_cost = (
        totals["input"] * PRICE_INPUT
        + totals["cache_write"] * PRICE_CACHE_WRITE
        + totals["cache_read"] * PRICE_CACHE_READ
        + totals["output"] * PRICE_OUTPUT
    ) / 1e6
    print(f"\nActual cost: ${actual_cost:.2f}")
    print(f"  Tokens: input={totals['input']:,} cache_read={totals['cache_read']:,} "
          f"cache_write={totals['cache_write']:,} output={totals['output']:,}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"voice_eval_{timestamp}.json"
    out_path.write_text(json.dumps({
        "summary": summary,
        "results": dict(results),
        "config": {"per_mode": args.per_mode, "seed": args.seed},
        "cost": actual_cost,
        "tokens": totals,
    }, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
