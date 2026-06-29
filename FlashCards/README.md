# Flashcards — Anki-Compatible Spaced Repetition

[← Back to Index](../INDEX.md)

---

## Card Index

| File | Phase | Cards | Estimated Review Time |
|------|-------|:-----:|:--------------------:|
| [Phase0_Flashcards.md](./Phase0_Flashcards.md) | Python + CS | 30 | 15 min |
| [Phase1_Flashcards.md](./Phase1_Flashcards.md) | Mathematics | 35 | 18 min |
| [Phase2_Flashcards.md](./Phase2_Flashcards.md) | Classical ML | 35 | 18 min |
| [Phase3_Flashcards.md](./Phase3_Flashcards.md) | DL + PyTorch | 40 | 20 min |
| [Phase4_Flashcards.md](./Phase4_Flashcards.md) | NLP + Transformers | 45 | 23 min |
| [Phase5_Flashcards.md](./Phase5_Flashcards.md) | LLMs + RAG | 45 | 23 min |
| [Phase6_Flashcards.md](./Phase6_Flashcards.md) | Agents + MCP | 35 | 18 min |
| [Phase7_Flashcards.md](./Phase7_Flashcards.md) | Fine-Tuning + RLHF | 35 | 18 min |
| [Phase8_Flashcards.md](./Phase8_Flashcards.md) | Advanced Topics | 30 | 15 min |
| [Phase9_Flashcards.md](./Phase9_Flashcards.md) | Production AI | 35 | 18 min |
| [Phase10_Flashcards.md](./Phase10_Flashcards.md) | Research + OSS | 25 | 13 min |
| **Total** | | **390** | **~3.3 hrs full review** |

---

## Card Format

Each card follows this exact structure:

```
---
**ID**: P0-001
**Front**: Question text
**Back**: Answer text
**Difficulty**: Beginner | Intermediate | Advanced
**Category**: Theory | Coding | Math | Production | Interview | Research
**Tags**: #phase0 #python #datatypes
---
```

---

## Categories Explained

| Category | Description | Example |
|----------|------------|---------|
| **Theory** | Conceptual understanding | "What is attention?" |
| **Coding** | Code patterns, APIs | "How do you enable 4-bit QLoRA?" |
| **Math** | Equations, derivations | "Write the LoRA update rule" |
| **Production** | Deployment, monitoring | "What metrics do you log per LLM call?" |
| **Interview** | Common interview questions | "Explain backpropagation to a PM" |
| **Research** | Papers, frontier knowledge | "What did InstructGPT prove?" |

---

## Spaced Repetition Schedule

| Review | When | Time Investment |
|--------|------|:--------------:|
| First review | Day of learning | — |
| Second review | 2 days later | 10 min/phase |
| Third review | 1 week later | 10 min/phase |
| Fourth review | 1 month later | 10 min/phase |
| Maintenance | Every 3 months | 5 min/phase |

---

## Exporting to Anki

Run this script to convert all flashcard files to Anki-importable TSV:

```python
"""
Convert Flashcards/*.md to Anki TSV format.
Usage: python export_anki.py
Output: anki_import.tsv (import into Anki via File > Import)
"""
import re
from pathlib import Path

def parse_cards(filepath: Path) -> list[dict]:
    content = filepath.read_text(encoding="utf-8")
    cards = []
    # Match card blocks between --- separators
    pattern = r"\*\*ID\*\*: (.+?)\n\*\*Front\*\*: (.+?)\n\*\*Back\*\*: (.+?)\n\*\*Difficulty\*\*: (.+?)\n\*\*Category\*\*: (.+?)\n\*\*Tags\*\*: (.+?)(?=\n---|\Z)"
    for m in re.finditer(pattern, content, re.DOTALL):
        cards.append({
            "id": m.group(1).strip(),
            "front": m.group(2).strip(),
            "back": m.group(3).strip().replace("\n", "<br>"),
            "difficulty": m.group(4).strip(),
            "category": m.group(5).strip(),
            "tags": m.group(6).strip(),
        })
    return cards

all_cards = []
for md_file in sorted(Path("Flashcards").glob("Phase*_Flashcards.md")):
    all_cards.extend(parse_cards(md_file))

with open("anki_import.tsv", "w", encoding="utf-8") as f:
    f.write("#separator:tab\n#html:true\n#notetype:Basic\n")
    for card in all_cards:
        front = f"[{card['id']}] {card['front']}<br><small>{card['tags']}</small>"
        back = card['back']
        f.write(f"{front}\t{back}\n")

print(f"Exported {len(all_cards)} cards to anki_import.tsv")
```

---

## How to Use

1. **During study**: after learning a concept, find its card and try to answer before revealing
2. **Daily review**: do 20–30 cards per day from phases you've completed
3. **Before interviews**: run all Interview-category cards
4. **Anki**: export and use automated spaced repetition
