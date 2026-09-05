#!/usr/bin/env python3
"""Matched capability evaluation for Gemma 4 E4B Q8 artifacts."""

from __future__ import annotations

from pathlib import Path

import gemma4_e2b_q8_capability as engine
import gemma4_e2b_q8_eval as gemma


ROOT = Path(__file__).resolve().parents[1]
gemma.TOKENIZER_PATH = ROOT / "checkpoints" / "gemma4-e4b-it"
gemma.SCHEMA_FAMILY = "gemma4-e4b"
engine.gemma = gemma
engine.RUN_DIR = ROOT / "runs" / "gemma4-e4b-q8" / "capability"
engine.SCHEMA_FAMILY = "gemma4-e4b"


if __name__ == "__main__":
    engine.main()
