#!/usr/bin/env python3
"""
Glitch-mode accuracy harness.

Asks Lanty's local GGUF model a battery of Linux/programming questions
(probes generated and auto-reviewed by Claude — see lanty_gen.probes)
and checks that each response contains the required keywords for the
correct answer.

Pass rate is the fraction of probes where every must_contain keyword
appears (case-insensitive) in Lanty's response.

Usage:
    pip install llama-cpp-python anthropic
    export ANTHROPIC_API_KEY=sk-ant-...   # only needed on first run / --regen
    python scripts/eval_glitch.py
    python scripts/eval_glitch.py --regen --per-topic 8
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from lanty_gen.probes import load_or_generate

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "lanty-qwen-Q4_K_M.gguf"
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "data" / "lanty_system_prompt.txt"
PROBE_CACHE = PROJECT_ROOT / "data" / "eval" / "glitch_probes.json"
EVAL_DIR = PROJECT_ROOT / "data" / "eval"


def keyword_match(response: str, keywords: list[str]) -> tuple[bool, list[str]]:
    """Returns (passed, missing). Case-insensitive substring match per keyword."""
    lower = response.lower()
    missing = [k for k in keywords if k.lower() not in lower]
    return len(missing) == 0, missing


def main():
    parser = argparse.ArgumentParser(description="Glitch-mode accuracy harness for Lanty")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to GGUF model")
    parser.add_argument("--probes", default=str(PROBE_CACHE), help="Probe cache path")
    parser.add_argument("--per-topic", type=int, default=5, help="Probes per topic when generating (default: 5)")
    parser.add_argument("--regen", action="store_true", help="Force probe regeneration via Claude (costs $$)")
    parser.add_argument("--ctx", type=int, default=2048, help="GGUF context size")
    parser.add_argument("--max-tokens", type=int, default=300, help="Max response tokens")
    args = parser.parse_args()

    # Probes (lazy-import anthropic only if regen is needed)
    cache_path = Path(args.probes)
    if args.regen or not cache_path.exists():
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: ANTHROPIC_API_KEY required for probe generation.")
            sys.exit(1)
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        probes = load_or_generate(client, cache_path, args.per_topic, regen=args.regen)
    else:
        probes = load_or_generate(client=None, cache_path=cache_path, per_topic=args.per_topic, regen=False)

    if not probes:
        print("No probes available. Aborting.")
        sys.exit(1)

    # Load Lanty's GGUF
    try:
        from llama_cpp import Llama
    except ImportError:
        print("Error: llama-cpp-python not installed. pip install llama-cpp-python")
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    system_prompt = SYSTEM_PROMPT_PATH.read_text().strip() if SYSTEM_PROMPT_PATH.exists() else ""

    print(f"\nLoading {model_path.name}...")
    llm = Llama(
        model_path=str(model_path),
        n_ctx=args.ctx,
        verbose=False,
    )
    print(f"Running {len(probes)} probes...\n")

    results = []
    passes = 0
    start = time.time()
    for i, probe in enumerate(probes, 1):
        out = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": probe["q"]},
            ],
            temperature=0.7,
            max_tokens=args.max_tokens,
        )
        response = out["choices"][0]["message"]["content"]
        passed, missing = keyword_match(response, probe["must_contain"])
        if passed:
            passes += 1
        results.append({
            "topic": probe.get("topic", "?"),
            "q": probe["q"],
            "must_contain": probe["must_contain"],
            "response": response,
            "passed": passed,
            "missing": missing,
        })
        marker = "PASS" if passed else "FAIL"
        print(f"  [{i:>3}/{len(probes)}] {marker} {probe['q'][:70]}")
        if not passed:
            print(f"          missing: {missing}")

    elapsed = time.time() - start
    rate = passes / len(probes) if probes else 0

    # Per-topic breakdown
    by_topic: dict = {}
    for r in results:
        bucket = by_topic.setdefault(r["topic"], {"pass": 0, "fail": 0})
        bucket["pass" if r["passed"] else "fail"] += 1

    print(f"\n=== Results ===")
    print(f"  Pass rate:  {passes}/{len(probes)}  ({rate:.1%})")
    print(f"  Elapsed:    {elapsed:.0f}s")
    print(f"\n  By topic:")
    for topic, b in sorted(by_topic.items()):
        total = b["pass"] + b["fail"]
        print(f"    {topic:<40} {b['pass']:>3}/{total:<3}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = EVAL_DIR / f"glitch_eval_{timestamp}.json"
    out_path.write_text(json.dumps({
        "pass_rate": rate,
        "passes": passes,
        "total": len(probes),
        "by_topic": by_topic,
        "results": results,
    }, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
