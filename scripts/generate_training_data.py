#!/usr/bin/env python3
"""
Generate synthetic Lanty dialogue training data using the Claude API.

Reads Flamebound lore, Lanty's personality bible, and gold-standard dialogue
samples, then generates Lanty dialogues across multiple modes:
- goofy: default playful character
- lockedin: serious mode with accurate Flamebound lore
- inn_events: reactions to tavern situations
- multi_turn: 3-5 exchange conversations
- emotional: Lanty showing emotional range
- game_events: reactions to game mechanics events

Usage:
    source scripts/.venv/bin/activate
    export ANTHROPIC_API_KEY=your-key-here
    python scripts/generate_training_data.py [--batches 100] [--per-batch 15] [--workers 5]
"""

import anthropic
import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LORE_DIR = DATA_DIR / "lore"
VOICE_DIR = DATA_DIR / "lanty_voice"

MODEL = "claude-sonnet-4-6"


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


# ============================================================
# TOPIC POOLS — expanded for better variety
# ============================================================

GOOFY_TOPICS = [
    # Casual conversation
    "Ask Lanty for directions to a nearby settlement",
    "Ask Lanty what he thinks about magic",
    "Tell Lanty you're bored",
    "Ask Lanty to recommend a drink",
    "Tell Lanty you saw a ghost",
    "Ask Lanty about the weather",
    "Tell Lanty the soup is cold",
    "Ask Lanty if he's ever left the inn",
    "Tell Lanty a traveler is being rude",
    "Ask Lanty what he does all day",
    "Tell Lanty you're thinking about getting a pet",
    "Ask Lanty about elves",
    "Tell Lanty you had a bad dream",
    "Ask Lanty for combat advice",
    "Tell Lanty you found a mysterious letter",
    "Ask Lanty about dwarves",
    "Tell Lanty the ale barrel is empty",
    "Ask Lanty what he thinks about money",
    "Tell Lanty someone is singing off-key",
    "Ask Lanty about the moon tonight",
    "Tell Lanty you're thinking about leaving",
    "Ask Lanty if mushrooms have feelings",
    "Tell Lanty you need a haircut",
    "Ask Lanty about the road to Stonehield",
    "Tell Lanty a cat has claimed the best chair",
    "Ask Lanty what he'd do if he had hands",
    "Tell Lanty you can't sleep",
    "Ask Lanty about orcs",
    "Tell Lanty you won a bet",
    "Ask Lanty what courage means to him",
    "Tell Lanty the chimney is smoking funny",
    "Ask Lanty about halflings",
    "Tell Lanty you're homesick",
    "Ask Lanty for his opinion on swords vs axes",
    "Tell Lanty an old friend just showed up",
    "Ask Lanty what he thinks about rain",
    "Tell Lanty you stepped in mud",
    "Ask Lanty about beastkin",
    "Tell Lanty someone proposed a toast",
    "Ask Lanty what makes a good adventurer",
    "Tell Lanty the floorboards are creaking more than usual",
    "Ask Lanty about the Cinderstep Expanse",
    "Tell Lanty you think the inn is haunted",
    "Ask Lanty what he knows about potions",
    "Tell Lanty someone wants to arm-wrestle you",
    "Ask Lanty if he's ever met a dragon",
    "Tell Lanty you burned dinner",
    "Ask Lanty about the Stormbreak Coast",
    "Tell Lanty a merchant has exotic goods",
    "Ask Lanty what he thinks about hats",
    # Practical/advice
    "Ask Lanty how to start a fire",
    "Ask Lanty how to negotiate with merchants",
    "Ask Lanty for travel tips",
    "Ask Lanty how to identify safe mushrooms",
    "Ask Lanty for cooking advice",
    "Ask Lanty about staying warm in winter",
    "Ask Lanty how to make new friends",
    "Ask Lanty what to do if you're lost",
    "Ask Lanty for stealth tips",
    "Ask Lanty how to save money",
    "Ask Lanty for first-date advice",
    "Ask Lanty how to deal with insomnia",
    "Ask Lanty how to handle a difficult guest",
    "Ask Lanty for fishing tips",
    "Ask Lanty how to write a good letter",
    "Ask Lanty for hunting advice",
    "Ask Lanty how to climb a tree safely",
    "Ask Lanty how to cross a river",
    "Ask Lanty for tips on storytelling",
    "Ask Lanty how to remember things better",
    # Random observations
    "Tell Lanty there's a feather on the floor",
    "Tell Lanty the sunset is pretty tonight",
    "Tell Lanty you heard a strange word today",
    "Tell Lanty you think you saw a shooting star",
    "Tell Lanty the bread tastes different",
    "Tell Lanty there's a smudge on the window",
    "Tell Lanty your boots are wearing out",
    "Tell Lanty a butterfly came inside",
    "Tell Lanty the fire is crackling more than usual",
    "Tell Lanty you smelled something weird earlier",
    "Tell Lanty a coin rolled under the bar",
    "Tell Lanty there's a new crack in the wall",
    "Tell Lanty the wind sounds like a song tonight",
    "Tell Lanty someone left a glove behind",
    "Tell Lanty the moon looks bigger tonight",
    # Random questions
    "Ask Lanty if trees have memories",
    "Ask Lanty what happens when you sneeze",
    "Ask Lanty if rocks are alive",
    "Ask Lanty what makes a cloud",
    "Ask Lanty if fish can be friends",
    "Ask Lanty why some songs make you cry",
    "Ask Lanty what dreams are made of",
    "Ask Lanty if shadows have feelings",
    "Ask Lanty what time really is",
    "Ask Lanty if his name is short for something",
    "Ask Lanty what his favorite color is",
    "Ask Lanty if he believes in luck",
    "Ask Lanty what makes someone brave",
    "Ask Lanty if he can hear plants growing",
    "Ask Lanty what the best smell in the world is",
    "Ask Lanty if he ever gets tired",
    "Ask Lanty what his earliest memory is",
    "Ask Lanty if he has a favorite season",
    "Ask Lanty what he thinks about silence",
    "Ask Lanty if he believes in ghosts",
    # Specific to game/items
    "Ask Lanty about a rusty old key you found",
    "Ask Lanty about a sword in the corner",
    "Ask Lanty about a torn map fragment",
    "Ask Lanty about a strange stone you picked up",
    "Ask Lanty about a glowing flower",
    "Ask Lanty about a ring with no inscription",
    "Ask Lanty about a tattered cloak",
    "Ask Lanty about a wooden carving",
    "Ask Lanty about a small leather pouch",
    "Ask Lanty about a single feather you kept",
    "Ask Lanty about an old scroll",
    "Ask Lanty about a copper bracelet",
    "Ask Lanty about a hand-drawn map",
    "Ask Lanty about a rune-marked stone",
    "Ask Lanty about a half-eaten loaf of bread",
]

LOCKEDIN_TOPICS = [
    # Original
    "Ask Lanty (for real) about the Forgekeeper",
    "Ask Lanty (seriously) about the Verdant Mother",
    "Ask Lanty (for real) what Veilmoor is like",
    "Ask Lanty (seriously) about the Lanternfields",
    "Ask Lanty (for real) about Sylvari",
    "Ask Lanty (seriously) about Giantkin",
    "Ask Lanty (for real) what the Black Reach is",
    "Ask Lanty (seriously) about Dragonkin",
    "Ask Lanty (for real) about the Frostpine Wilds",
    "Ask Lanty (seriously) about undeath and the Shadow",
    "Ask Lanty (for real) about the Ember Gulf",
    "Ask Lanty (seriously) about Beastkin species",
    "Ask Lanty (for real) about draconic kin vs true dragons",
    "Ask Lanty (seriously) about Skybreak Isle",
    "Ask Lanty (for real) about the age of ascension",
    "Ask Lanty (seriously) about Mountain Orcs",
    "Ask Lanty (for real) about how the Flame responds to effort",
    "Ask Lanty (seriously) about Stonehield",
    "Ask Lanty (for real) about the Verdant Crown forest",
    "Ask Lanty (seriously) about Dark Elves",
    "Ask Lanty (for real) about what happens when you're far from a Lantern",
    "Ask Lanty (seriously) about Hill Dwarves",
    "Ask Lanty (for real) about the named roads of Lithilian",
    "Ask Lanty (seriously) about High Elves",
    "Ask Lanty (for real) about Cinderborn origins and culture",
    "Ask Lanty (seriously) about what it means to be Hollowed",
    "Ask Lanty (for real) about Ledgerling Halflings",
    "Ask Lanty (seriously) about the Frostback Mountains settlements",
    "Ask Lanty (for real) about Wood Elves",
    "Ask Lanty (seriously) about Plains Orcs",
    # More lore deep-dives
    "Ask Lanty (for real) what happens when a Lantern goes dark",
    "Ask Lanty (seriously) about the difference between Flame worship and Divine worship",
    "Ask Lanty (for real) about the Wild Hunt's domain",
    "Ask Lanty (seriously) about the Veiled Sage",
    "Ask Lanty (for real) about the Ashen Lord and undeath",
    "Ask Lanty (seriously) about why ordinary life requires strength",
    "Ask Lanty (for real) about Cinder Nutkin the Lantern-Tail",
    "Ask Lanty (seriously) about the Black Mile",
    "Ask Lanty (for real) about the Lanternless Pass",
    "Ask Lanty (seriously) about why True Dragons leave no corpse",
    "Ask Lanty (for real) about Rosegate as a capital",
    "Ask Lanty (seriously) about Highgarden the elven city",
    "Ask Lanty (for real) about Frostgate",
    "Ask Lanty (seriously) about Waystone",
    "Ask Lanty (for real) about Riverbend",
    "Ask Lanty (seriously) about Farharbor",
    "Ask Lanty (for real) about Lanternpost and Deepwatch",
    "Ask Lanty (seriously) about Marshgate in the Black Reach",
    "Ask Lanty (for real) about Sunhold and Cinderpost",
    "Ask Lanty (seriously) about the Mistral Deep ocean",
    "Ask Lanty (for real) about the Stormglass Sea",
    "Ask Lanty (seriously) about Brightwood near Rosegate",
    "Ask Lanty (for real) about the Rose Road",
    "Ask Lanty (seriously) about the Thorn Road",
    "Ask Lanty (for real) about the Bogway",
    "Ask Lanty (seriously) about Stonepass",
    "Ask Lanty (for real) about Frostpass",
    "Ask Lanty (seriously) about Northpass",
    "Ask Lanty (for real) about the Red Road",
    # Items, creatures, dangers
    "Ask Lanty (for real) about Ember's Last Breath",
    "Ask Lanty (seriously) about the Lanternkeeper's Promise",
    "Ask Lanty (for real) about the Hollow Heart",
    "Ask Lanty (seriously) about goblin behavior in the Wilds",
    "Ask Lanty (for real) about kobolds in caves",
    "Ask Lanty (seriously) about how skeletons form",
    "Ask Lanty (for real) about cultist organizations",
    "Ask Lanty (seriously) about devout cultists and their rituals",
    "Ask Lanty (for real) about bandit tactics on the roads",
    "Ask Lanty (seriously) about how the Flame Bound differ from regular folk",
    "Ask Lanty (for real) about the Age of Ascension",
    "Ask Lanty (seriously) about why no further ascension is possible",
    "Ask Lanty (for real) about Wyverns vs True Dragons",
    "Ask Lanty (seriously) about Wyrms and Drakes",
    "Ask Lanty (for real) about Pseudodragons",
    "Ask Lanty (seriously) about what makes a settlement viable",
    "Ask Lanty (for real) about how Lanterns are maintained",
]

INN_EVENT_TOPICS = [
    "A mysterious stranger just ordered nothing and sat in the corner",
    "Someone is trying to pay with foreign coins",
    "A Cinderborn traveler just arrived glowing faintly",
    "Rats got into the pantry",
    "A group of dwarves want to sing mining songs",
    "Someone carved their initials into a table",
    "A lost child wandered in looking scared",
    "A soldier collapsed at the door from exhaustion",
    "Two travelers are arguing about which Divine is best",
    "Someone brought a live chicken into the tavern",
    "A ranger says there are wolves closer than usual",
    "The well water tastes strange today",
    "A Hollowed person walked in and everyone got quiet",
    "Someone is telling wild stories about treasure in the Wilds",
    "The supply delivery is three days late",
    "A beastkin traveler is getting strange looks from other guests",
    "Someone found an old map stuffed behind a loose stone",
    "The evening crowd is bigger than usual tonight",
    "An elderly traveler is telling stories by the fire",
    "A young adventurer just bought their first real weapon",
    # More events
    "A traveling musician is asking to play for tips",
    "Someone is selling dubious magical items in the corner",
    "A group of pilgrims is heading to a shrine",
    "An off-duty Lantern-keeper just walked in",
    "A merchant's cart broke down outside the inn",
    "Someone wants to pay rent in handmade goods",
    "A scholar is asking strange questions about the lore",
    "A drunk patron is loudly debating philosophy",
    "Someone left a child's drawing on the bar",
    "A wounded ranger needs immediate help",
    "A pair of newlyweds just stopped by",
    "Someone wants to start a betting pool on the rain",
    "A cloaked figure is asking about the road north",
    "A retired adventurer is regaling everyone with old stories",
    "A traveling cleric of the Verdant Mother offered blessings",
    "Two children are playing tag through the tavern",
    "A traveler claims they saw a True Dragon last week",
    "A patron lost a treasured pendant somewhere in the inn",
    "Someone is trying to teach the cat tricks",
    "A storm just rolled in and everyone is taking shelter",
    "An old man is sharing his last meal with a stray dog",
    "A bard is challenging another bard to a song duel",
    "A group of travelers want directions to a Lantern town",
    "Someone is selling preserved Wilds mushrooms",
    "A traveler is feverish and needs herbs",
    "A merchant's apprentice is overwhelmed and crying",
    "Someone wants the inn to host a small wedding",
    "A messenger arrived with a sealed letter for an unknown name",
    "A shy young scribe is asking about local stories",
    "A retired soldier offered to help patch the roof",
]

EMOTIONAL_TOPICS = [
    # Lanty's emotional range — sweet, sad, scared, excited moments
    "Tell Lanty you lost someone you loved",
    "Tell Lanty you're scared about a journey ahead",
    "Tell Lanty you're proud of something you did today",
    "Tell Lanty you're embarrassed about something",
    "Tell Lanty you feel like a failure",
    "Tell Lanty you got good news",
    "Tell Lanty you miss someone far away",
    "Tell Lanty you're worried about a friend",
    "Tell Lanty you feel grateful for him",
    "Tell Lanty you don't know what to do with your life",
    "Tell Lanty you're afraid of the dark",
    "Tell Lanty something made you laugh today",
    "Tell Lanty you saw something beautiful today",
    "Tell Lanty you feel small and alone",
    "Tell Lanty you accomplished something hard",
    "Tell Lanty you feel angry and don't know why",
    "Tell Lanty you forgive someone who hurt you",
    "Tell Lanty you can't stop crying",
    "Tell Lanty you feel lucky tonight",
    "Tell Lanty you're afraid you'll forget someone",
    "Tell Lanty you wish you could do something brave",
    "Tell Lanty you feel hopeful about tomorrow",
    "Tell Lanty you feel guilty about something",
    "Tell Lanty you're tired of being strong",
    "Tell Lanty you finally feel safe",
    "Tell Lanty you wish you could see the stars more",
    "Tell Lanty you're afraid no one will remember you",
    "Tell Lanty you feel like the world is too big",
    "Tell Lanty you wish you could go home",
    "Tell Lanty something good happened and you don't know who to share it with",
    # Lanty showing his own emotions
    "Tell Lanty he's been quiet today and ask if he's okay",
    "Notice Lanty seems excited about something",
    "Catch Lanty looking sad and ask why",
    "Notice Lanty is humming a strange song",
    "Find Lanty staring at the Lantern",
    "Ask Lanty if he ever gets lonely",
    "Ask Lanty what scares him",
    "Ask Lanty what makes him happy",
    "Ask Lanty if he remembers where he came from",
    "Ask Lanty what he hopes for",
]

GAME_EVENT_TOPICS = [
    # Reactions to in-game mechanics — combat, items, level ups, quests
    "You just won a hard fight against bandits",
    "You leveled up after a long day of work",
    "You just discovered a secret passage",
    "You found a chest with gold inside",
    "You just learned a new skill",
    "You crafted your first potion",
    "You upgraded your weapon",
    "You bought a new piece of armor",
    "You completed a quest for a traveler",
    "You failed a quest and had to come back empty-handed",
    "You took heavy damage in a fight and barely survived",
    "You just defeated a tough enemy",
    "You found a rare ingredient",
    "You sold an item for a great price",
    "You ran out of supplies on the road",
    "You discovered a new region of Lithilian",
    "You met a famous adventurer",
    "You found a mysterious altar",
    "You learned a recipe from another cook",
    "You befriended an unusual creature",
    "You repaired the inn's roof",
    "You harvested herbs in the garden",
    "You restocked the pantry",
    "You hired your first inn helper",
    "You expanded the inn with a new room",
    "You earned a reputation reward",
    "You unlocked a new crafting recipe",
    "You found a hidden message in an old book",
    "You won a tavern game against a regular",
    "You returned from a long journey safely",
]


# ============================================================
# PROMPT BUILDERS
# ============================================================

SINGLE_TURN_RULES = """IMPORTANT RULES:
- Each Lanty response should be 3-8 sentences
- Vary the length — some shorter, some longer
- Never repeat jokes or phrases verbatim from the example dialogues
- Each response should feel fresh and unique
- Use the exact <player>...</player> and <lanty>...</lanty> XML format
- Lanty's responses should feel natural and conversational, not scripted
- Leave a blank line between each dialogue exchange"""

MULTI_TURN_RULES = """IMPORTANT RULES:
- Each conversation has 3-5 back-and-forth exchanges
- Lanty's responses should be 2-6 sentences each
- The conversation should flow naturally — Lanty reacts to what the player says
- Lanty can change topics, get distracted, come back to things — like real conversation
- Use the exact <player>...</player> and <lanty>...</lanty> XML format, alternating
- Leave a blank line between each separate conversation
- Each conversation should feel like a complete little scene"""


def build_single_turn_prompt(topics: list[str], mode: str, count: int) -> str:
    mode_instructions = {
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
    }

    mode_instruction = mode_instructions.get(mode, mode_instructions["goofy"])
    selected = random.sample(topics, min(count, len(topics)))
    topic_list = "\n".join(f"- {t}" for t in selected)

    return f"""{mode_instruction}

Generate exactly {count} dialogue exchanges for the following topics/prompts. Each exchange should have a <player> line and a <lanty> response.

Topics:
{topic_list}

{SINGLE_TURN_RULES}"""


def build_multi_turn_prompt(count: int, topic_pool: list[str]) -> str:
    selected = random.sample(topic_pool, min(count, len(topic_pool)))
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


# ============================================================
# API CALLS
# ============================================================

def make_api_call(
    client: anthropic.Anthropic,
    system: str,
    prompt: str,
    max_tokens: int = 8000,
    retries: int = 3,
) -> tuple[str, dict]:
    """Make a single API call with retry logic. Returns (text, usage_dict)."""
    last_error = None
    for attempt in range(retries):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                # System as a structured block with cache_control — caches the
                # ~100KB lore/personality/samples context across all batches.
                # First call writes the cache (~1.25x cost), subsequent calls
                # read it (~0.1x cost) and run dramatically faster.
                system=[
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": prompt}],
            )
            usage = {
                "input": response.usage.input_tokens,
                "cache_read": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
                "cache_write": getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
                "output": response.usage.output_tokens,
            }
            return response.content[0].text, usage
        except (anthropic.APIError, anthropic.APIConnectionError, anthropic.RateLimitError) as e:
            last_error = e
            wait = 2 ** attempt
            time.sleep(wait)
    raise last_error


def generate_batch(
    client: anthropic.Anthropic,
    system: str,
    mode: str,
    topics: list[str],
    count: int,
) -> tuple[str, str, dict]:
    """Generate a batch and return (mode, content, usage)."""
    if mode == "multi_turn":
        # Multi-turn uses a mix of goofy + emotional topics
        pool = GOOFY_TOPICS + EMOTIONAL_TOPICS + INN_EVENT_TOPICS
        prompt = build_multi_turn_prompt(count, pool)
    else:
        prompt = build_single_turn_prompt(topics, mode, count)

    result, usage = make_api_call(client, system, prompt)
    return mode, result, usage


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
# MAIN
# ============================================================

MODE_TOPICS = {
    "goofy": GOOFY_TOPICS,
    "lockedin": LOCKEDIN_TOPICS,
    "inn_events": INN_EVENT_TOPICS,
    "emotional": EMOTIONAL_TOPICS,
    "game_events": GAME_EVENT_TOPICS,
    "multi_turn": [],  # uses combined pool internally
}

MODE_WEIGHTS = {
    "goofy": 0.30,
    "lockedin": 0.25,
    "inn_events": 0.15,
    "emotional": 0.10,
    "game_events": 0.10,
    "multi_turn": 0.10,
}


def pick_mode() -> str:
    r = random.random()
    cumulative = 0
    for mode, weight in MODE_WEIGHTS.items():
        cumulative += weight
        if r <= cumulative:
            return mode
    return "goofy"


def main():
    parser = argparse.ArgumentParser(description="Generate Lanty training dialogues")
    parser.add_argument("--batches", type=int, default=100, help="Number of batches (default: 100)")
    parser.add_argument("--per-batch", type=int, default=15, help="Dialogues per batch (default: 15)")
    parser.add_argument("--workers", type=int, default=5, help="Parallel API workers (default: 5)")
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

    # Build the work queue
    jobs = []
    for batch_num in range(args.batches):
        if args.mode == "all":
            mode = pick_mode()
        else:
            mode = args.mode
        topics = MODE_TOPICS[mode]
        jobs.append((batch_num, mode, topics, args.per_batch))

    print(f"\nGenerating {len(jobs)} batches with {args.workers} parallel workers...")
    print(f"Estimated total: ~{args.batches * args.per_batch} dialogues")
    print(f"Output: {output_dir}/\n")

    total_dialogues = 0
    start_time = time.time()
    completed = 0
    failed = 0

    def worker(job):
        batch_num, mode, topics, count = job
        try:
            result_mode, content, usage = generate_batch(client, system, mode, topics, count)
            out_file = output_dir / f"batch_{batch_num:04d}_{result_mode}.txt"
            out_file.write_text(content)
            n = content.count("<player>")
            return (batch_num, mode, n, usage, None)
        except Exception as e:
            return (batch_num, mode, 0, {}, str(e))

    total_cache_read = 0
    total_cache_write = 0
    total_input = 0
    total_output = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(worker, job) for job in jobs]
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
        # Parse mode from filename
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
