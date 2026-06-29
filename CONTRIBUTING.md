# Contributing Guide

> This repository is a personal learning curriculum. Contributions that improve clarity, fix errors, or add genuinely valuable content are welcome.

---

## What to Contribute

**High value**:
- Corrections to factual errors, outdated information, or broken code
- Missing explanations for concepts referenced but not explained
- Additional interview questions with model answers
- New flashcards covering gaps in existing sets
- Curated resource additions (must meet the quality bar below)

**Medium value**:
- Improved code examples (clearer, more Pythonic, better comments)
- Additional "common mistakes" entries
- Better visualizations or diagrams

**Low value / declined**:
- Stylistic rewrites that don't add information
- Duplicate content already covered elsewhere in the curriculum
- Resources without proper metadata (difficulty, free/paid, etc.)

---

## Quality Bar

Before submitting, check:

1. **Accurate** — Code runs. Math is correct. Claims are sourced.
2. **Concise** — No padding. Every sentence earns its place.
3. **Non-duplicate** — Search existing files first with `grep -r "keyword" .`
4. **Consistent format** — Match the formatting conventions in the file you're editing.

---

## File Naming Conventions

| Type | Convention | Example |
|------|-----------|---------|
| Phase files | `##_PhaseN_Name.md` | `07_Phase6_AI_Agents.md` |
| CheatSheets | `PhaseN_Topic.md` | `Phase4_NLP_Transformers.md` |
| Flashcards | `PhaseN_Flashcards.md` | `Phase4_Flashcards.md` |
| Resources | `TopicName.md` | `ResearchPapers.md` |

---

## Code Style

All Python code in this repo follows:

```python
# Imports at top, grouped: stdlib → third-party → local
import os
from pathlib import Path

import numpy as np
import torch

# Type hints on function signatures
def embed_text(text: str, model: str = "text-embedding-3-small") -> list[float]:
    ...

# No bare except. No eval(). No hardcoded secrets.
# f-strings preferred over .format()
# NumPy vectorized ops preferred over Python loops
```

**Security rules** (mandatory):
- Never use `eval()` or `exec()` on user input — use `simpleeval` or AST parsing
- Never hardcode API keys — use `os.environ` or `.env` + python-dotenv
- Input validation at system boundaries

---

## Flashcard Format

Every flashcard must follow this exact format:

```markdown
---
**ID**: P0-001
**Front**: What is the difference between list and tuple in Python?
**Back**: List is mutable (can change after creation); tuple is immutable. Tuples are faster to iterate and can be used as dict keys.
**Difficulty**: Beginner
**Category**: Theory
**Tags**: #phase0 #python #datatypes
---
```

- `ID`: Phase prefix + 3-digit number (e.g., `P4-023`)
- `Difficulty`: `Beginner` | `Intermediate` | `Advanced`
- `Category`: `Theory` | `Coding` | `Math` | `Production` | `Interview` | `Research`

---

## Resource Entry Format

Every resource in the `Resources/` folder must include:

```markdown
### Resource Name

| Field | Value |
|-------|-------|
| **URL** | https://... |
| **Why useful** | One sentence |
| **Difficulty** | Beginner / Intermediate / Advanced |
| **Prerequisites** | What to know first |
| **Best time to use** | Which phase to read/watch this |
| **Estimated time** | X hours / X weeks |
| **Free/Paid** | Free / Paid ($XX) / Freemium |
| **Actively maintained** | Yes / No (last updated YYYY) |
```

---

## Pull Request Process

1. Fork the repo and create a branch: `fix/phase4-attention-typo` or `add/phase5-flashcards`
2. Make targeted changes. One PR per logical change.
3. Ensure all code in your PR actually runs
4. Update the relevant section of `CHANGELOG.md`
5. PR description: what you changed + why

---

## Reporting Issues

Open a GitHub Issue for:
- Factual errors with a citation to the correct information
- Broken code with the error message
- Missing prerequisite — "Phase X assumes Y but Y is not explained"

---

*[← Index](./INDEX.md)*
