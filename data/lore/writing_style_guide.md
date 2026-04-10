# 📝 Forge Link Campaign Writing Style Guide (V5)

This document defines the standard formatting and writing style for all campaign content in Forge Link. V5 reflects the evolved, practical style seen in the Dragonslayer Vault and Apothecary content.

---

## 🔄 Changes from V4

| Change | Rationale |
|--------|-----------|
| **Simplified TOC format** | Full nested TOC at document start for easy navigation |
| **Quick Reference tables** | Time, Goal, Threat summary for each room at a glance |
| **Flexible POI structure** | Tables for locations, prose for deep dives |
| **Integrated dialogue** | NPC reactions embedded in relevant sections |
| **Q&A blocks for NPCs** | "Questions Players Might Ask" tables |
| **Combat in dedicated section** | Separate from POI when combat is significant |
| **Session flow markers** | "What Happens Next" and "Session End" guidance |
| **Removed rigid numbering** | Section headers by name, not letter.number |

---

## 📋 Document Structure

### Top-Level Header

```markdown
# 🔮 Lower Level X — [Level Name]

---
```

Use emoji prefix matching content type:
- 🔮 Magical/mystical areas
- 🧪 Laboratories/alchemy
- ⚔️ Combat-focused
- 🏛️ Archives/lore
- 🔥 Fire/forge themes
- 🌲 Outdoor/travel

---

## 📜 Table of Contents

Full nested structure at document start:

```markdown
## 📜 Table of Contents

### 📜 Overview & History
- First Impressions
- The [Location]'s Past
- What Changed

### 🏛️ Room A: [Name]
- First Impression
- Atmosphere & Details
- Points of Interest
- [Significant POI Name]
- [Another POI Name]

### 🔥 Room B: [Name]
- [Same pattern]

### ⚔️ Encounters & Combat
- [If significant combat exists]

### 🎲 DM Notes
- Session flow guidance
```

---

## 📐 Room Structure

### Quick Reference Table

Start each room with a quick reference:

```markdown
## 📜 Quick Reference

| ⏱️ Time | 🎯 Goal | ⚔️ Threat |
|---------|---------|-----------|
| ~30-45 min | Lore + Exploration | None — eerie peace |
```

---

### First Impression (Read-Aloud)

```markdown
### 👁️ First Impression

> *Read-aloud boxed text describing what players see, hear, smell upon entry. Use evocative language. Highlight key visual elements with {color:text} formatting.*
>
> *Second paragraph if needed.*
```

**Color formatting:**
- `{gold:text}` — Important items, magical glow
- `{green:text}` — Positive, safety, nature
- `{fire:text}` — Fire, danger (Eldereth references)
- `{red:text}` — Threat, warning
- `{danger:text}` — Mechanical danger
- `{purple:text}` — Magic items

---

### Atmosphere & Details

```markdown
### 🌡️ Atmosphere & Details

- **Lighting:** Description
- **Sound:** What they hear. {green:Emotional note if relevant.}
- **Smell:** Scents present
- **Temperature:** Physical feeling
- **Feeling:** Emotional atmosphere
```

---

### Points of Interest Table

Quick reference for what's in the room:

```markdown
### 🔍 Points of Interest

| Location | What They Find |
|----------|----------------|
| 📖 **Central Feature** | Brief description |
| 📚 **Secondary Feature** | Brief description |
| 🚪 **Exit** | Where it leads |
```

---

### Detailed POI Sections

Each significant element gets its own section with prose:

```markdown
## 📖 [POI Name]

> *Optional italicized flavor text*

**What It Is:** Summary paragraph explaining function and importance.

### [Subsection as needed]

> *Read aloud text if appropriate*

[Prose description with mechanics embedded]

**Specific Entry on [Subject]:**

*"Quoted text from documents or inscriptions in italics"*

- **Bullet:** Detail
- **Bullet:** Detail
- {green:Insight:} Highlighted information

**Mechanical Benefit (Optional):** What players gain from this.
```

---

## 💬 NPC & Dialogue Formatting

### Named NPC Dialogue

```markdown
### 💬 [NPC Name] at [Location]

> *[NPC Name] [action description].*

**[NPC Name]:** *"Dialogue in italics with quotes."*

*[Description of their action or expression.]*

**[NPC Name]:** *"More dialogue."*
```

### Questions Players Might Ask

```markdown
### 💬 Questions Players Might Ask

| Question | [NPC]'s Answer |
|----------|----------------|
| **"Question text?"** | *"Answer in italics"* |
| **"Another question?"** | *"Another answer"* |
```

---

## ⚔️ Combat Section

### Enemy Quick Stats

```markdown
### ⚔️ The Enemy

#### 🔥 {red:[Boss Name]} — [Title]

> *Italicized description of appearance and personality.*

| Stat | Value |
|------|-------|
| **HP** | XX |
| **AC** | XX (armor type) |
| **Speed** | XX ft |
| **Attacks** | +X to hit, damage |
| **Spellcasting** | DC XX, spell slots |

**Key Spells/Abilities:**
- {fire:Ability Name} — effect
- {danger:Ability Name} — effect

**Tactics:**
- Bullet list of combat behavior
- When they use specific abilities
- Retreat or death conditions

**Personality:**
- Combat dialogue tendencies
- If bloodied: *"Dialogue"*
```

### Combat Flow

```markdown
### ⚔️ Combat Flow

**Round 1 — [Phase Name]**
- What happens
- Key decision points

**Round 2-3 — [Phase Name]**
- Development
- Environmental use

**Round 4+ — [Phase Name]**
- Escalation or desperation moves

**Ending the Fight:**
- Victory conditions
- Fleeing enemies
```

### Mid-Fight Dialogue

```markdown
### 🎭 Mid-Fight Dialogue

> *Use these when dramatically appropriate:*

**[NPC] (start):** *"Dialogue"*

**[NPC] (bloodied):** *"Dialogue"*

**[NPC] (dying):** *"Dialogue"*
```

---

## 🏆 Treasure & Loot

```markdown
### 🏆 Loot

| Item | Location |
|------|----------|
| {purple:Item Name} | Where found |
| {gold:Currency amount} | Where found |
| Potion of X (quantity) | Where found |
| Document/Clue | Where found |
```

For magic items, add detail:

```markdown
### ⚔️ [Weapon Category]

| Weapon | Type | Properties |
|--------|------|------------|
| 🏹 {purple:Name} | Type | +X bonus. Special effect. |
| 🗡️ {purple:Name} | Type | +X bonus. Special effect. |
```

---

## 🎯 Session Flow Sections

### What Happens Next

```markdown
### 🎯 What Happens Next

| Choice | Outcome |
|--------|---------|
| **Option A?** | Result and consequences |
| **Option B?** | Result and consequences |
| **Option C?** | Result and consequences |
```

### If Things Go Wrong

```markdown
### 💀 If Things Go Wrong

> *Backup options if the fight is too hard:*

- **[Intervention]:** Description
- **[Environmental option]:** Description
- **[Escape route]:** Description
```

### Session End

```markdown
### 🎬 Session End

> **Read aloud:**

> *"Closing narration in italics. Summarize the moment. End on a dramatic beat.*
>
> *{gold:Location Name} awaits.*
>
> ***End of session.***"
```

---

## 🗺️ Travel Sections

For overland/dungeon travel between major locations:

```markdown
## 🗺️ The Road to [Destination]

> *Quick reference for travel:*

| Detail | Info |
|--------|------|
| **Direction** | Cardinal direction and landmarks |
| **Distance** | Miles or hours |
| **Travel Time** | {gold:X days} on foot |
| **Terrain** | Description |
| **Encounters** | Optional suggestions |
| **Arrival** | When and in what state |
```

---

## 🎨 Formatting Standards

### Color Tags

| Tag | Use Case | Example |
|-----|----------|---------|
| `{gold:text}` | Important items, magic, hope | {gold:The Emberfall Codex} |
| `{green:text}` | Safety, nature, positive | {green:Peaceful. Reverent.} |
| `{fire:text}` | Eldereth, flames, heat | {fire:Eldereth Firestorm} |
| `{red:text}` | Threats, warnings, enemy | {red:A shape.} |
| `{danger:text}` | Mechanical hazards | {danger:difficult terrain} |
| `{purple:text}` | Magic items, artifacts | {purple:Wyrmseeker} |

### Emoji Usage

**Section Headers:**
- 📜 Overview, lore, documents
- 👁️ First impressions
- 🌡️ Atmosphere
- 🔍 Points of interest
- 💬 Dialogue, Q&A
- ⚔️ Combat, enemies
- 🏆 Treasure, loot
- 🎯 Next steps
- 🎬 Session markers
- 🗺️ Travel, maps
- 🎲 DM Notes

**Inline Markers:**
- 🔥 Fire, heat, forge
- 💎 Treasure
- 🚪 Doors, exits
- 👑 Bosses
- 🐉 Dragons

### Tables vs Prose

**Use tables for:**
- Quick reference data
- POI lists
- Enemy stats
- Loot summaries
- Q&A blocks
- Choice/outcome matrices

**Use prose for:**
- Read-aloud text
- Detailed POI exploration
- Combat flow narration
- NPC dialogue scenes
- Atmospheric description

### Dialogue Formatting

```markdown
**[Name]:** *"Dialogue in italics with quotes."*

> *Action or emotional beat in block quote.*

**[Name]:** *"Continued dialogue."*
```

---

## ✅ V5 Checklist

When writing content, ensure:

- [ ] Quick Reference table at room/section start
- [ ] First Impression read-aloud in blockquote
- [ ] Atmosphere & Details bullet list
- [ ] Points of Interest table for overview
- [ ] Detailed sections for significant POIs
- [ ] NPC dialogue with name labels
- [ ] Q&A table for talkative NPCs
- [ ] Combat section with stat tables and flow phases
- [ ] Loot table with locations
- [ ] "What Happens Next" for decision points
- [ ] Session End narration for chapter closes
- [ ] Color formatting for emphasis
- [ ] Consistent emoji usage

---

## 📏 Length Guidelines

Based on actual content analysis:

| Section Type | Approximate Length |
|--------------|-------------------|
| **Quick Reference** | 3-4 line table |
| **First Impression** | 2-4 paragraph blockquote |
| **Atmosphere** | 5-6 bullet points |
| **POI Table** | 4-8 rows |
| **Detailed POI** | 100-300 words |
| **NPC Q&A** | 5-10 questions |
| **Combat Section** | 300-600 words |
| **Full Room** | 500-1500 words |
| **Full Level** | 2000-4000 words |

---

## 🔄 Migration from V4

1. **Add Quick Reference tables** — Time, Goal, Threat
2. **Convert rigid numbering** — Use descriptive headers
3. **Add color formatting** — {gold:}, {fire:}, etc.
4. **Add Q&A blocks** — For significant NPCs
5. **Add session flow markers** — What Happens Next, Session End
6. **Consolidate DM Notes** — Into footer or inline tips

---

*V5 reflects the evolved style from the Dragonslayer Vault and Apothecary content, prioritizing readability, DM usability, and dramatic session pacing.*
