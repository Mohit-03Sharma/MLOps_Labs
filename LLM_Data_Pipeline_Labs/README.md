# LLM Data Pipeline — Lab 2

A streaming language modeling data pipeline built with Hugging Face `datasets` and PyTorch.

---

## What this lab does

Demonstrates how to build a **true streaming LLM pipeline** that:

- Loads a large text corpus without reading it all into RAM
- Tokenizes text on the fly
- Uses a rolling buffer to concatenate and chunk tokens into fixed-length blocks
- Wraps everything in a PyTorch `IterableDataset` + `DataLoader` ready for model training

---

## Notebook structure

| Cell | Description |
|---|---|
| 1 | Docstring — goals and teaching points |
| 2 | Package installs (commented out) |
| 3 | Imports |
| 4 | Load `roneneldan/TinyStories` in streaming mode |
| 5 | Initialize `bert-base-uncased` tokenizer |
| 6 | Lazy tokenization via `.map()` |
| 7 | Rolling buffer generator for fixed-length blocks |
| 8 | `StreamingLMIterableDataset` wrapper |
| 9 | Collate function |
| 10 | DataLoader setup |
| 11 | Shape verification over 3 batches |
| 12 | Top-10 token-ID frequency analysis |
| 13 | Decode first block back to readable text |

---

## Setup

```bash
pip install transformers datasets torch
```

Then open and run `Lab2_modified.ipynb` top to bottom.

> **Note:** Data streams directly from Hugging Face Hub — no local download required.

---

## Requirements

- Python 3.9+
- `transformers >= 4.30`
- `datasets >= 2.14`
- `torch >= 2.0`