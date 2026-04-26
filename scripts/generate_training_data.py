#!/usr/bin/env python3
"""
Generate synthetic Lanty dialogue training data using the Claude API.

Reads Flamebound lore, Lanty's personality bible, and gold-standard dialogue
samples, then generates Lanty dialogues across multiple modes:
- goofy:       default playful character
- lockedin:    serious mode with accurate Flamebound lore
- inn_events:  reactions to tavern situations
- multi_turn:  3-5 exchange conversations
- emotional:   Lanty showing emotional range
- game_events: reactions to game mechanics events
- glitch:      Lanty accidentally knowing Linux/programming

The heavy prompt context (~100KB) is cached via Anthropic prompt caching;
see lanty_gen/api.py.

Usage:
    source scripts/.venv/bin/activate
    export ANTHROPIC_API_KEY=your-key-here
    python scripts/generate_training_data.py [--batches 100] [--per-batch 15] [--workers 3]
"""

import argparse
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

from lanty_gen.api import generate_batch
from lanty_gen.prompts import (
    DATA_DIR,
    VOICE_DIR,
    build_system_prompt,
    load_gold_samples,
    load_lore_context,
    load_personality,
)
from lanty_gen.topics import MODE_TOPICS

# Weights control how many batches each mode gets when --mode=all.
# Must sum to 1.0 (not enforced, but pick_mode() falls back to goofy if short).
MODE_WEIGHTS = {
    "goofy": 0.25,
    "lockedin": 0.20,
    "inn_events": 0.12,
    "emotional": 0.08,
    "game_events": 0.08,
    "glitch": 0.15,
    "multi_turn": 0.12,
}


def pick_mode() -> str:
    r = random.random()
    cumulative = 0
    for mode, weight in MODE_WEIGHTS.items():
        cumulative += weight
        if r <= cumulative:
            return mode
    return "goofy"


BATCH_PATTERN = re.compile(r"^batch_(\d{4})_[a-z_]+\.txt$")


def plan_batch_numbers(output_dir: Path, count: int, no_resume: bool) -> tuple[list[int], int, int]:
    """Decide which batch numbers to use for `count` new batches.

    Default: fill gaps in the existing 0000..max range first, then append
    above the current max. Keeps numbering contiguous over time as crashed
    or pruned batches get backfilled.

    --no-resume: skip gap-filling and only append after max_existing. Never
    overwrites existing files (would destroy training data).

    Returns (batch_numbers, gaps_filled, new_appended).
    """
    existing = set()
    if output_dir.exists():
        for f in output_dir.glob("batch_*.txt"):
            m = BATCH_PATTERN.match(f.name)
            if m:
                existing.add(int(m.group(1)))

    if not existing:
        return list(range(count)), 0, count

    max_existing = max(existing)

    if no_resume:
        new = list(range(max_existing + 1, max_existing + 1 + count))
        return new, 0, count

    gaps = sorted(set(range(max_existing + 1)) - existing)
    fill = gaps[:count]
    remaining = count - len(fill)
    new = list(range(max_existing + 1, max_existing + 1 + remaining))

    return fill + new, len(fill), remaining


def run_worker(client, system, job, output_dir):
    batch_num, mode, topics, count = job
    try:
        result_mode, content, usage = generate_batch(client, system, mode, topics, count)
        out_file = output_dir / f"batch_{batch_num:04d}_{result_mode}.txt"
        out_file.write_text(content)
        n = content.count("<player>")
        return (batch_num, mode, n, usage, None)
    except Exception as e:
        return (batch_num, mode, 0, {}, str(e))


def main():
    parser = argparse.ArgumentParser(description="Generate Lanty training dialogues")
    parser.add_argument("--batches", type=int, default=100, help="Number of batches (default: 100)")
    parser.add_argument("--per-batch", type=int, default=15, help="Dialogues per batch (default: 15)")
    parser.add_argument("--workers", type=int, default=3, help="Parallel API workers (default: 3, safe for Tier 1 rate limits)")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DATA_DIR / "training"),
        help="Output directory",
    )
    parser.add_argument(
        "--mode",
        choices=list(MODE_TOPICS.keys()) + ["all"],
        default="all",
        help="Which dialogue mode to generate (default: all)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Skip gap-filling, only append new batches after the highest existing number. Never overwrites.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading lore and personality data...")
    lore = load_lore_context()
    personality = load_personality()
    samples = load_gold_samples()
    print(f"  Lore: {len(lore):,} chars")
    print(f"  Personality: {len(personality):,} chars")
    print(f"  Gold samples: {len(samples):,} chars")

    system = build_system_prompt(personality, samples, lore)

    batch_numbers, gaps_filled, new_appended = plan_batch_numbers(
        output_dir, args.batches, args.no_resume
    )
    jobs = []
    for batch_num in batch_numbers:
        mode = pick_mode() if args.mode == "all" else args.mode
        topics = MODE_TOPICS[mode]
        jobs.append((batch_num, mode, topics, args.per_batch))

    print(f"\nGenerating {len(jobs)} batches with {args.workers} parallel workers...")
    if not args.no_resume and gaps_filled > 0:
        print(f"  Resume mode: filling {gaps_filled} gap(s), appending {new_appended} new")
    print(f"Estimated total: ~{args.batches * args.per_batch} dialogues")
    print(f"Output: {output_dir}/\n")

    total_dialogues = 0
    total_cache_read = 0
    total_cache_write = 0
    total_input = 0
    total_output = 0
    start_time = time.time()
    completed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_worker, client, system, job, output_dir) for job in jobs]
        for future in as_completed(futures):
            batch_num, mode, n, usage, error = future.result()
            completed += 1
            if error:
                failed += 1
                print(f"  [{completed}/{len(jobs)}] batch {batch_num:04d} {mode}: FAILED - {error}")
            else:
                total_dialogues += n
                total_cache_read += usage.get("cache_read", 0)
                total_cache_write += usage.get("cache_write", 0)
                total_input += usage.get("input", 0)
                total_output += usage.get("output", 0)
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(jobs) - completed) / rate if rate > 0 else 0
                cache_indicator = "HIT" if usage.get("cache_read", 0) > 0 else "MISS"
                print(
                    f"  [{completed}/{len(jobs)}] batch {batch_num:04d} {mode}: {n} dialogues "
                    f"[cache {cache_indicator}] ({rate:.2f} batch/s, ETA {eta:.0f}s)"
                )

    # Combine all batches by mode
    print(f"\nCombining batches...")
    by_mode = {}
    for f in sorted(output_dir.glob("batch_*.txt")):
        parts = f.stem.split("_", 2)
        if len(parts) >= 3:
            mode = parts[2]
            by_mode.setdefault(mode, []).append(f.read_text().strip())

    for mode, contents in by_mode.items():
        combined_path = output_dir / f"combined_{mode}.txt"
        combined_path.write_text("\n\n".join(contents))
        n = sum(c.count("<player>") for c in contents)
        print(f"  combined_{mode}.txt: {n} dialogues")

    # Copy gold samples into training dir for completeness
    for f in VOICE_DIR.glob("dialogues_*.txt"):
        dest = output_dir / f"gold_{f.name}"
        dest.write_text(f.read_text())

    elapsed = time.time() - start_time
    print(f"\n=== Done! ===")
    print(f"  Total dialogues generated: {total_dialogues}")
    print(f"  Failed batches:            {failed}")
    print(f"  Time elapsed:              {elapsed:.0f}s")
    print(f"  Output directory:          {output_dir}")
    print(f"\n  Token usage:")
    print(f"    Input (uncached): {total_input:,}")
    print(f"    Cache writes:     {total_cache_write:,}")
    print(f"    Cache reads:      {total_cache_read:,}")
    print(f"    Output:           {total_output:,}")
    if total_cache_read > 0:
        cache_total = total_cache_read + total_cache_write + total_input
        cache_pct = 100 * total_cache_read / cache_total if cache_total > 0 else 0
        print(f"    Cache hit rate:   {cache_pct:.1f}% of input tokens served from cache")


if __name__ == "__main__":
    main()
