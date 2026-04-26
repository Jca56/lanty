"""Generate and auto-review glitch-mode accuracy probes.

A "probe" is a Linux/programming question paired with a list of keywords
or concepts a correct answer MUST contain (case-insensitive substring
match). We generate them with Claude, then have a SECOND Claude pass
review each one for technical accuracy. Probes that fail review are
dropped — no human in the loop.

Cached on disk at data/eval/glitch_probes.json. Re-run with --regen to
rebuild from scratch.
"""

import json
import re
import time
from pathlib import Path

import anthropic

MODEL = "claude-sonnet-4-6"

PROBE_TOPICS = [
    "Linux kernel basics",
    "Arch Linux package management with pacman",
    "shell features (bash, zsh, fish)",
    "git version control",
    "file system concepts",
    "process management and signals",
    "networking primitives (TCP, DNS, ports)",
    "compilers vs interpreters",
    "Rust language features",
    "Python language features",
    "memory management and pointers",
    "common Linux commands",
    "systemd service management",
    "tmux and terminal multiplexers",
    "package managers (npm, cargo, pip)",
    "regex syntax basics",
    "JSON and data formats",
    "ssh and remote shell",
    "filesystems and inodes",
    "version control branching and merging",
]


def build_generator_prompt(topic: str, count: int) -> str:
    return f"""Generate {count} accuracy probes for the topic: "{topic}".

Each probe is a question a Linux/programming-savvy person would answer
correctly, paired with a list of keywords/concepts that MUST appear
(case-insensitive substring match) in any correct answer.

Rules:
- Questions should be SHORT — under 15 words.
- Keywords should be the MINIMUM set that proves the answer is right.
  Don't list everything that COULD appear; list what MUST appear.
- Prefer concept words over command names. "synchronize" + "upgrade" is
  better than "pacman -Syu" because the model might phrase it differently.
- If a question has multiple correct phrasings, list keywords for the
  CORE concept only.

Return ONLY a JSON array, no commentary:
[
  {{"q": "what does pacman -Syu do?", "must_contain": ["sync", "upgrade"]}},
  {{"q": "what is a daemon?", "must_contain": ["background", "process"]}}
]"""


REVIEW_PROMPT = """Review the following accuracy probe for a Linux/programming
quiz. Your job: decide if the keywords are MINIMAL and CORRECT for the
question.

PASS: keywords are accurate AND minimal — every keyword is something
that must appear in any correct answer, and there are no missing
critical concepts.

FAIL: a keyword is wrong, too broad, or there's a missing critical
keyword that any correct answer would need.

Probe:
{probe_json}

Respond ONLY with a JSON object: {{"verdict": "pass" or "fail", "reason": "..."}}"""


def _extract_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    return json.loads(text)


def generate_probes(client: anthropic.Anthropic, per_topic: int = 5) -> list[dict]:
    """Generate raw probes across all topics. Returns list of {q, must_contain}."""
    all_probes: list[dict] = []
    for topic in PROBE_TOPICS:
        prompt = build_generator_prompt(topic, per_topic)
        response = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        try:
            probes = _extract_json(response.content[0].text)
            for p in probes:
                p["topic"] = topic
            all_probes.extend(probes)
            print(f"  generated {len(probes):>2} probes for: {topic}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  FAILED to parse probes for {topic}: {e}")
        time.sleep(0.3)  # gentle on rate limits
    return all_probes


def review_probes(client: anthropic.Anthropic, probes: list[dict]) -> tuple[list[dict], list[dict]]:
    """Auto-review each probe with a second Claude pass. Returns (passed, failed)."""
    passed: list[dict] = []
    failed: list[dict] = []
    for i, probe in enumerate(probes):
        prompt = REVIEW_PROMPT.format(probe_json=json.dumps({
            "q": probe["q"],
            "must_contain": probe["must_contain"],
        }))
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            verdict = _extract_json(response.content[0].text)
            probe["_review"] = verdict
            if verdict.get("verdict") == "pass":
                passed.append(probe)
            else:
                failed.append(probe)
        except Exception as e:
            probe["_review"] = {"verdict": "error", "reason": str(e)}
            failed.append(probe)
        if (i + 1) % 10 == 0:
            print(f"  reviewed {i + 1}/{len(probes)} (pass={len(passed)} fail={len(failed)})")
        time.sleep(0.2)
    return passed, failed


def load_or_generate(client: anthropic.Anthropic, cache_path: Path, per_topic: int, regen: bool) -> list[dict]:
    """Load reviewed probes from cache or generate+review fresh ones."""
    if cache_path.exists() and not regen:
        data = json.loads(cache_path.read_text())
        probes = data.get("passed", [])
        print(f"Loaded {len(probes)} cached probes from {cache_path}")
        return probes

    print(f"Generating probes ({per_topic}/topic across {len(PROBE_TOPICS)} topics)...")
    raw = generate_probes(client, per_topic=per_topic)
    print(f"\nGenerated {len(raw)} raw probes. Auto-reviewing...")
    passed, failed = review_probes(client, raw)
    print(f"\nReview complete: {len(passed)} passed, {len(failed)} failed")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({
        "passed": passed,
        "failed": failed,
        "config": {"per_topic": per_topic, "topics": PROBE_TOPICS},
    }, indent=2))
    print(f"Cached to {cache_path}")
    return passed
