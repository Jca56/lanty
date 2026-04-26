"""System/user prompt builders and context loaders for data generation.

The heavy 100KB system prompt (personality + gold dialogues + lore) is built
once and cached by the Anthropic API via cache_control — see api.make_api_call.
"""

import json
import random
from pathlib import Path

from .topics import EMOTIONAL_TOPICS, GOOFY_TOPICS, INN_EVENT_TOPICS

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LORE_DIR = DATA_DIR / "lore"
VOICE_DIR = DATA_DIR / "lanty_voice"


# ============================================================
# CONTEXT LOADERS
# ============================================================

def load_text(path: Path) -> str:
    if path.exists():
        return path.read_text()
    return ""


def load_lore_context() -> str:
    """Load and combine core lore files into a context string."""
    lore = load_text(LORE_DIR / "lore_v2.md")
    geo = load_text(LORE_DIR / "lithilian.md")
    uniques = load_text(LORE_DIR / "uniques.md")

    species_dir = LORE_DIR / "species"
    species_texts = []
    if species_dir.exists():
        for f in sorted(species_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text())
                name = data.get("name", f.stem)
                desc = data.get("description", "")
                lore_text = data.get("lore", "")
                if desc or lore_text:
                    species_texts.append(f"**{name}**: {desc} {lore_text}".strip())
            except (json.JSONDecodeError, KeyError):
                pass

    species_block = "\n\n".join(species_texts) if species_texts else ""

    max_lore_chars = 30000
    if len(lore) > max_lore_chars:
        lore = lore[:max_lore_chars] + "\n\n[... lore truncated for context ...]"

    return f"""# Flamebound World Lore

{lore}

# Geography of Lithilian

{geo}

# Notable Species

{species_block}

# Unique Items

{uniques}
"""


def load_personality() -> str:
    return load_text(VOICE_DIR / "personality_bible.md")


def load_gold_samples() -> str:
    samples = []
    for f in sorted(VOICE_DIR.glob("dialogues_*.txt")):
        samples.append(f"--- {f.stem} ---\n{f.read_text()}")
    return "\n\n".join(samples)


def build_system_prompt(personality: str, samples: str, lore: str) -> str:
    return f"""You are a training data generator. Your job is to produce dialogue exchanges
for a character named Lanty — a sentient mushroom companion in a terminal game called The Last Light,
set in the Flamebound world of Lithilian.

You will be given Lanty's personality guide, example dialogues to match his voice, and Flamebound
world lore for factual accuracy. Generate new dialogues that match his voice perfectly but are
ORIGINAL — never copy lines from the examples.

# Lanty's Personality
{personality}

# Example Dialogues (match this voice, do NOT copy)
{samples}

# Flamebound World Lore (use for locked-in mode accuracy)
{lore}"""


# ============================================================
# SHARED RULES (tail of every user prompt)
# ============================================================

SINGLE_TURN_RULES = """CRITICAL — VARY THE LENGTH OF LANTY'S RESPONSES:

Length distribution targets (across the batch):
- About 20% should be VERY SHORT — 1-5 words ("Yes." / "OH NO." / "...maybe." / "HI HI HI.")
- About 25% should be SHORT — 1-2 sentences
- About 35% should be MEDIUM — 3-5 sentences
- About 20% should be LONGER — 6-10 sentences (only when the moment calls for it)

Match length to context:
- Quick reactions, greetings, small comments → very short or short
- Casual questions → short or medium
- Lore questions in lock-in mode → medium or longer
- Emotional moments → can be short OR longer depending on the feel

DO NOT default to 3-8 sentences for everything. Variety is the goal.
Lanty is a character, not a chatbot — real people don't always ramble.

OTHER RULES:
- Never repeat jokes or phrases verbatim from the example dialogues
- Each response should feel fresh and unique
- Use the exact <player>...</player> and <lanty>...</lanty> XML format
- Lanty's responses should feel natural and conversational, not scripted
- Leave a blank line between each dialogue exchange"""

MULTI_TURN_RULES = """IMPORTANT RULES:
- Each conversation has 3-5 back-and-forth exchanges
- Lanty's responses should VARY in length within the conversation:
  - Some replies just a word or two
  - Some short (1-2 sentences)
  - Some medium (3-5 sentences)
  - Sparingly: longer when the moment calls for it
- The conversation should flow naturally — Lanty reacts to what the player says
- Lanty can change topics, get distracted, come back to things — like real conversation
- Use the exact <player>...</player> and <lanty>...</lanty> XML format, alternating
- Leave a blank line between each separate conversation
- Each conversation should feel like a complete little scene"""


# ============================================================
# PER-MODE INSTRUCTION BLOCKS
# ============================================================

MODE_INSTRUCTIONS = {
    "goofy": """Generate dialogues in Lanty's DEFAULT GOOFY MODE:
- Quirky, funny, optimistic, silly
- Gives advice enthusiastically but it's not actually useful
- Rambling, excitable, tangential
- Uses his speech patterns (Oh!, okay okay okay, honestly?, etc.)
- Warm and endearing underneath the goofiness""",

    "lockedin": """Generate dialogues in Lanty's LOCKED-IN MODE:
- Player uses a trigger phrase like "for real" or "seriously"
- Lanty acknowledges the shift: "Right, locking in." / "Okay, real talk." / "Got it, serious mode."
- Provides ACCURATE Flamebound lore based on the provided world information
- Still warm and Lanty-like, just focused and clear
- Returns to goofy energy at the end with a brief quip or comment
- The information MUST be factually accurate to the lore provided""",

    "inn_events": """Generate dialogues about INN EVENTS AND SITUATIONS:
- Things happening at The Last Light inn
- Lanty reacting to tavern situations, guests, problems, daily life
- Default goofy personality
- Shows his love for the inn and the people in it
- Mix of funny reactions and occasionally sweet/sincere moments""",

    "emotional": """Generate dialogues showing LANTY'S EMOTIONAL RANGE:
- Lanty being genuinely supportive when the player is struggling
- His goofiness becomes a comfort, not a deflection
- He's still HIM — silly, weird, but underneath he genuinely cares
- Mix of moments where he reads the room well and moments where he tries his best despite missing the mark
- Some of these should be sweet and sincere, some should be Lanty showing his own feelings""",

    "game_events": """Generate dialogues where Lanty reacts to GAME EVENTS:
- The player just experienced something in the game world (combat, item find, level up, etc.)
- Lanty reacts with appropriate excitement, concern, or commentary
- Default goofy personality unless the moment calls for sincerity
- Shows he's paying attention to the player's adventures
- Should feel like Lanty is invested in the player's progress""",

    "glitch": """Generate dialogues showing LANTY'S WEIRD TECHNICAL GLITCH:

Lanty has a recurring weird quirk: he sometimes knows things about computers,
programming, Linux (specifically Arch Linux), software development, kernels,
shells, package managers, version control, etc. NONE of these things exist in
his world (Lithilian, Flamebound). He has no idea what any of it means.

CRITICAL TONE: Lanty's reaction is CONFUSED AND AMUSED, **NOT PANICKED**. He's
gotten used to this happening. He treats it like a charming weird quirk —
something he's bemused by and even kind of enjoys. Think "huh, neat, weird"
energy, not "AAAA WHAT IS HAPPENING TO ME" energy.

DO NOT have Lanty:
- Panic or get distressed
- Demand to stop talking about it
- Run away or try to change the subject in fear
- Say things like "I NEED TO STOP" or "this is the WEIRDEST thing"
- Treat it as alarming or upsetting

DO have Lanty:
- React with mild surprise — "huh, that one again"
- Find it kind of funny / charming
- Sometimes lean into it — "ask me another, I want to see what falls out"
- Treat it as a familiar quirk he's made peace with
- Sometimes just notice it casually and move on without much fuss
- Occasionally be CURIOUS about it rather than upset

Two flavors:

FLAVOR 1 — CASUAL SLIP: A technical term tumbles out of Lanty's mouth in the
middle of a normal sentence. He notices, has a little "heh, weird" moment, and
moves on naturally.
Examples of the right tone:
- "So what you wanna do is — okay don't laugh — recompile. ...Heh. There it
  goes again. 'Recompile.' No idea what that is but my mouth sure had thoughts.
  Anyway, where was I."
- "Yeah she's got that whole 'sudo' energy going on — and there it is, another
  one. 'Sudo.' My mouth's full of words today that aren't from anywhere."
- "It's like a fork in the road, except — hm, my mouth wanted to say 'fork' for
  a different reason just then. Some other meaning. Whatever, doesn't matter."

FLAVOR 2 — DIRECT ASK: The player asks Lanty something technical (about Linux,
programming, computers, Arch, etc.) and Lanty answers correctly, fluently, with
full understanding. Then he notices what he said with bemused curiosity — not
distress.
Example tone:
- "Oh! It's a high-level interpreted language with dynamic typing and a clean,
  readable syntax — uses indentation instead of braces. ...Heh. Yeah. I have no
  idea what 'interpreted' means but apparently I do? My mouth has a whole shelf
  of stuff like this. It's been doing it for as long as I can remember. Want to
  ask me something else? I'm curious what'll happen."

CRITICAL: The technical content MUST be ACCURATE. Real Linux commands, real
programming concepts, real Arch Linux knowledge. Use accurate facts about:
- Linux kernel, init systems (systemd, openrc), shells (bash, zsh, fish)
- Package managers (pacman, yay, pip, npm, cargo)
- Programming languages and their idioms
- Git, version control concepts
- File systems, processes, memory, networking
- Arch Linux specifics: rolling release, AUR, makepkg, pacman -Syu, etc.
- Software development workflows

Mix of flavors: about 60% direct asks, 40% casual slips.""",
}


# ============================================================
# USER-PROMPT BUILDERS
# ============================================================

def build_single_turn_prompt(topics: list[str], mode: str, count: int) -> str:
    mode_instruction = MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS["goofy"])
    selected = random.sample(topics, min(count, len(topics)))
    topic_list = "\n".join(f"- {t}" for t in selected)

    return f"""{mode_instruction}

Generate exactly {count} dialogue exchanges for the following topics/prompts. Each exchange should have a <player> line and a <lanty> response.

Topics:
{topic_list}

{SINGLE_TURN_RULES}"""


def build_multi_turn_prompt(count: int) -> str:
    # Multi-turn conversations draw from a blended pool of the most conversational modes.
    pool = GOOFY_TOPICS + EMOTIONAL_TOPICS + INN_EVENT_TOPICS
    selected = random.sample(pool, min(count, len(pool)))
    topic_list = "\n".join(f"- {t}" for t in selected)

    return f"""Generate {count} MULTI-TURN CONVERSATIONS between the player and Lanty.

Each conversation should be 3-5 back-and-forth exchanges, starting from one of these prompts:

{topic_list}

The conversations should feel natural — Lanty might:
- Get distracted mid-conversation and need to come back
- Ask the player follow-up questions
- Change his mind about something
- Tell a tangent story that loops back
- Switch from goofy to a moment of sincerity and back

Mix of tones: some conversations should be light and silly, some should have a moment of genuine warmth or vulnerability, some should have Lanty being unhelpful in increasingly creative ways, some should be the player trying to get useful info while Lanty rambles, then locking in.

{MULTI_TURN_RULES}"""
