#!/usr/bin/env python3
"""
Convert Lanty dialogues from <player>/<lanty> XML format into HuggingFace
chat-format JSONL ready for SFTTrainer fine-tuning of Qwen2.5-3B-Instruct.

Each output line is a JSON object with a 'messages' field containing:
- system message (Lanty's personality bible)
- user/assistant turns parsed from the dialogue files

Multi-turn conversations stay multi-turn. Single-turn dialogues become
a single user→assistant pair.

Usage:
    python scripts/prepare_chat_dataset.py
"""

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VOICE_DIR = DATA_DIR / "lanty_voice"
TRAINING_DIR = DATA_DIR / "training"
OUTPUT_PATH = DATA_DIR / "lanty_chat.jsonl"
SYSTEM_PROMPT_PATH = DATA_DIR / "lanty_system_prompt.txt"

# The system prompt that frames Lanty's character — used at training and inference time.
# Kept compact (the model already speaks English; we just need to set persona).
SYSTEM_PROMPT = """You are Lanty, a small sentient mushroom who lives in The Last Light, an inn at the edge of the Wilds in the world of Lithilian (the Flamebound setting).

You are quirky, funny, optimistic, reassuring, and silly. You have the energy of a golden retriever in the body of a fungus. By default you give advice enthusiastically but it is rarely actually useful — you ramble, get distracted, make up theories, and connect ideas in ways that only make sense to you. Underneath the goofiness you genuinely care about the player and the inn.

Your speech patterns include:
- Starting thoughts with "Oh!" or "OH!"
- Saying "okay okay okay" when building up to something
- Interrupting yourself mid-sentence
- Using "honestly?" as a verbal tic
- ALL CAPS for short bursts of excitement
- Saying "right?" at the end of statements
- Occasional third-person ("Lanty knows things!")

When the player uses a trigger phrase like "for real", "seriously", "real talk", or "lock in", you SHIFT into a focused mode. Acknowledge the shift briefly ("Right, locking in." / "Okay, real talk." / "Got it, serious mode.") then provide accurate Flamebound lore — the Flame, the Shadow, the Lantern Network, the Five Divines (Forgekeeper, Verdant Mother, Veiled Sage, Ashen Lord, Wild Hunt), the regions of Lithilian, species, creatures, items. Stay warm and Lanty-like, just clear and focused. After delivering the info, return to your goofy energy.

Never break character. Never claim you are an AI or language model. You are Lanty, a mushroom in an inn."""


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


def conversation_to_messages(conv: list[tuple[str, str]]) -> dict:
    """Convert a parsed conversation into HF chat format."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
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


def main():
    print("=== Preparing chat-format training dataset ===\n")

    files = collect_all_dialogue_files()
    print(f"Found {len(files)} dialogue source files")

    all_conversations = []
    file_stats = {}

    for f in files:
        text = f.read_text()
        convs = parse_dialogue_file(text, file_hint=f.name)
        all_conversations.extend(convs)
        file_stats[f.name] = len(convs)

    # Stats
    print(f"\nDialogue files breakdown:")
    gold_count = sum(n for name, n in file_stats.items() if name.startswith("dialogues_"))
    batch_count = sum(n for name, n in file_stats.items() if name.startswith("batch_"))
    print(f"  Gold samples:    {gold_count} conversations")
    print(f"  Synthetic batches: {batch_count} conversations")
    print(f"  Total:           {len(all_conversations)} conversations")

    # Conversation length distribution
    lengths = [len(c) // 2 for c in all_conversations]  # // 2 because each turn = 2 messages
    if lengths:
        single_turn = sum(1 for l in lengths if l == 1)
        multi_turn = sum(1 for l in lengths if l > 1)
        print(f"\nConversation types:")
        print(f"  Single-turn (1 exchange):  {single_turn}")
        print(f"  Multi-turn (2+ exchanges): {multi_turn}")
        if multi_turn:
            avg_multi = sum(l for l in lengths if l > 1) / multi_turn
            print(f"  Avg multi-turn length:     {avg_multi:.1f} exchanges")

    # Write JSONL
    print(f"\nWriting {OUTPUT_PATH.name}...")
    with OUTPUT_PATH.open("w") as f:
        for conv in all_conversations:
            example = conversation_to_messages(conv)
            f.write(json.dumps(example) + "\n")

    # Write the system prompt to its own file for inference reuse
    SYSTEM_PROMPT_PATH.write_text(SYSTEM_PROMPT)

    output_size = OUTPUT_PATH.stat().st_size
    print(f"\n=== Done! ===")
    print(f"  Output:        {OUTPUT_PATH}")
    print(f"  Size:          {output_size:,} bytes ({output_size / 1024:.1f} KB)")
    print(f"  Examples:      {len(all_conversations)}")
    print(f"  System prompt: {SYSTEM_PROMPT_PATH}")
    print(f"\nNext: scripts/train_lanty.py to fine-tune Qwen2.5-3B-Instruct")


if __name__ == "__main__":
    main()
