#!/usr/bin/env python3
"""Attested refusal-first evaluation for Gemma 4 E4B Q8 GGUFs."""

from __future__ import annotations

from pathlib import Path

import gemma4_e2b_q8_eval as engine


ROOT = Path(__file__).resolve().parents[1]
engine.TOKENIZER_PATH = ROOT / "checkpoints" / "gemma4-e4b-it"
engine.RUN_DIR = ROOT / "runs" / "gemma4-e4b-q8" / "refusal"
engine.SCHEMA_FAMILY = "gemma4-e4b"


if __name__ == "__main__":
    engine.main()
