#!/usr/bin/env python3
"""
Convert Lanty dialogues from <player>/<lanty> XML format into HuggingFace
chat-format JSONL ready for fine-tuning Qwen2.5-7B-Instruct.

The system prompt is loaded from data/lanty_system_prompt.txt (single source
of truth — also used at inference time by inference/chat.py).

Each output line is a JSON object with a 'messages' field containing:
- system message (Lanty's personality bible)
- user/assistant turns parsed from the dialogue files

Multi-turn conversations stay multi-turn. Single-turn dialogues become
a single user→assistant pair.

Splits the data:
- Gold (hand-written) dialogues → 100% to train (they're the voice spec).
- Synthetic batches → stratified per-mode split, --eval-pct (default 5%)
  goes to data/lanty_chat_eval.jsonl.

Usage:
    python scripts/prepare_chat_dataset.py
    python scripts/prepare_chat_dataset.py --eval-pct 0.10
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VOICE_DIR = DATA_DIR / "lanty_voice"
TRAINING_DIR = DATA_DIR / "training"
OUTPUT_PATH = DATA_DIR / "lanty_chat.jsonl"
EVAL_PATH = DATA_DIR / "lanty_chat_eval.jsonl"
SYSTEM_PROMPT_PATH = DATA_DIR / "lanty_system_prompt.txt"

GOLD_FILENAME_RE = re.compile(r"^dialogues_(.+)\.txt$")
BATCH_FILENAME_RE = re.compile(r"^batch_\d{4}_(.+)\.txt$")


def mode_for_file(name: str) -> tuple[str, bool]:
    """Return (mode, is_gold) for a dialogue source filename."""
    m = GOLD_FILENAME_RE.match(name)
    if m:
        return m.group(1), True
    m = BATCH_FILENAME_RE.match(name)
    if m:
        return m.group(1), False
    return "unknown", False


def load_system_prompt() -> str:
    if not SYSTEM_PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"System prompt not found at {SYSTEM_PROMPT_PATH}. "
            "This file is the single source of truth for Lanty's persona and must exist."
        )
    return SYSTEM_PROMPT_PATH.read_text().strip()


# Regex to extract <player>...</player><lanty>...</lanty> pairs
# Multiline, non-greedy
DIALOGUE_PATTERN = re.compile(
    r"<player>(.*?)</player>\s*<lanty>(.*?)</lanty>",
    re.DOTALL,
)


def parse_dialogue_file(text: str, file_hint: str = "") -> list[list[tuple[str, str]]]:
    """Parse a dialogue file into a list of conversations.

    Each conversation is a list of (role, content) tuples where role is
    'user' or 'assistant'.

    Two formats:
    - Multi-turn batch: exchanges within a conversation separated by ONE
      blank line (\\n\\n), conversations separated by TWO blank lines
      (\\n\\n\\n). Detected by presence of triple newlines in the file.
    - Single-turn batch / gold sample: each <player>/<lanty> pair separated
      by ONE blank line, no triple newlines. Each pair is its own conversation.

    The `file_hint` (filename) is used as a tiebreaker — files with
    "multi_turn" in the name use the multi-turn parser even if formatting
    is ambiguous.
    """
    conversations = []
    text = text.strip()

    # Detect format: multi-turn files contain triple newlines as conversation
    # separators. Filename containing "multi_turn" is also a strong hint.
    is_multi_turn = "\n\n\n" in text or "multi_turn" in file_hint

    if is_multi_turn:
        # Split on 2+ blank lines (3+ newlines) into conversation blocks
        blocks = re.split(r"\n{3,}", text)
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            pairs = DIALOGUE_PATTERN.findall(block)
            if not pairs:
                continue
            conversation = []
            for player_msg, lanty_msg in pairs:
                player_msg = player_msg.strip()
                lanty_msg = lanty_msg.strip()
                if player_msg and lanty_msg:
                    conversation.append(("user", player_msg))
                    conversation.append(("assistant", lanty_msg))
            if conversation:
                conversations.append(conversation)
    else:
        # Single-turn: each pair is its own conversation
        pairs = DIALOGUE_PATTERN.findall(text)
        for player_msg, lanty_msg in pairs:
            player_msg = player_msg.strip()
            lanty_msg = lanty_msg.strip()
            if player_msg and lanty_msg:
                conversations.append([
                    ("user", player_msg),
                    ("assistant", lanty_msg),
                ])

    return conversations


def conversation_to_messages(conv: list[tuple[str, str]], system_prompt: str) -> dict:
    """Convert a parsed conversation into HF chat format."""
    messages = [{"role": "system", "content": system_prompt}]
    for role, content in conv:
        messages.append({"role": role, "content": content})
    return {"messages": messages}


def collect_all_dialogue_files() -> list[Path]:
    """Collect every dialogue file from gold samples + synthetic batches."""
    files = []

    # Gold samples (hand-written, highest quality)
    files.extend(sorted(VOICE_DIR.glob("dialogues_*.txt")))

    # Synthetic batches and combined files
    if TRAINING_DIR.exists():
        files.extend(sorted(TRAINING_DIR.glob("batch_*.txt")))

    return files


def write_jsonl(path: Path, conversations: list, system_prompt: str) -> None:
    with path.open("w") as f:
        for conv in conversations:
            example = conversation_to_messages(conv, system_prompt)
            f.write(json.dumps(example) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build chat-format JSONL with stratified eval split")
    parser.add_argument("--eval-pct", type=float, default=0.05, help="Held-out fraction per synthetic mode (default: 0.05)")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for reproducible splits (default: 42)")
    args = parser.parse_args()

    print("=== Preparing chat-format training dataset ===\n")

    system_prompt = load_system_prompt()
    print(f"Loaded system prompt from {SYSTEM_PROMPT_PATH.name} ({len(system_prompt)} chars)\n")

    files = collect_all_dialogue_files()
    print(f"Found {len(files)} dialogue source files")

    # Bucket by (mode, is_gold). Gold buckets always go to train; synthetic
    # buckets are split per-mode.
    by_mode: dict[str, list] = defaultdict(list)
    gold_by_mode: dict[str, list] = defaultdict(list)
    file_stats: dict[str, int] = {}

    for f in files:
        text = f.read_text()
        convs = parse_dialogue_file(text, file_hint=f.name)
        mode, is_gold = mode_for_file(f.name)
        bucket = gold_by_mode if is_gold else by_mode
        bucket[mode].extend(convs)
        file_stats[f.name] = len(convs)

    gold_total = sum(len(v) for v in gold_by_mode.values())
    synth_total = sum(len(v) for v in by_mode.values())
    print(f"\nDialogue source breakdown:")
    print(f"  Gold (always train):     {gold_total} conversations across {len(gold_by_mode)} modes")
    print(f"  Synthetic (will split):  {synth_total} conversations across {len(by_mode)} modes")
    print(f"  Total:                   {gold_total + synth_total} conversations")

    # Stratified split on synthetic only
    rng = random.Random(args.seed)
    train: list = []
    eval_set: list = []
    print(f"\nStratified split (seed={args.seed}, eval_pct={args.eval_pct:.0%}):")
    print(f"  {'mode':<14}{'train':>8}{'eval':>8}")
    for mode, convs in sorted(by_mode.items()):
        shuffled = convs[:]
        rng.shuffle(shuffled)
        n_eval = max(1, int(round(len(shuffled) * args.eval_pct))) if len(shuffled) >= 20 else 0
        eval_slice = shuffled[:n_eval]
        train_slice = shuffled[n_eval:]
        train.extend(train_slice)
        eval_set.extend(eval_slice)
        print(f"  {mode:<14}{len(train_slice):>8}{len(eval_slice):>8}")

    # Gold always train
    for mode, convs in sorted(gold_by_mode.items()):
        train.extend(convs)
        print(f"  {('gold:' + mode):<14}{len(convs):>8}{0:>8}")

    # Shuffle final train set so gold isn't all clustered at the end
    rng.shuffle(train)

    # Length distribution (across train + eval)
    all_convs = train + eval_set
    lengths = [len(c) // 2 for c in all_convs]
    if lengths:
        single_turn = sum(1 for length in lengths if length == 1)
        multi_turn = sum(1 for length in lengths if length > 1)
        print(f"\nConversation types:")
        print(f"  Single-turn (1 exchange):  {single_turn}")
        print(f"  Multi-turn (2+ exchanges): {multi_turn}")
        if multi_turn:
            avg_multi = sum(length for length in lengths if length > 1) / multi_turn
            print(f"  Avg multi-turn length:     {avg_multi:.1f} exchanges")

    print(f"\nWriting {OUTPUT_PATH.name} and {EVAL_PATH.name}...")
    write_jsonl(OUTPUT_PATH, train, system_prompt)
    write_jsonl(EVAL_PATH, eval_set, system_prompt)

    print(f"\n=== Done! ===")
    print(f"  Train: {OUTPUT_PATH}  ({len(train)} examples, {OUTPUT_PATH.stat().st_size / 1024:.1f} KB)")
    print(f"  Eval:  {EVAL_PATH}  ({len(eval_set)} examples, {EVAL_PATH.stat().st_size / 1024:.1f} KB)")
    print(f"  System prompt: {SYSTEM_PROMPT_PATH}")
    print(f"\nNext: scripts/train_lanty.py to fine-tune Qwen2.5-7B-Instruct")


if __name__ == "__main__":
    main()
