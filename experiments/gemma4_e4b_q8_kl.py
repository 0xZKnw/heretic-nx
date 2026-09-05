#!/usr/bin/env python3
"""Exact full-vocabulary first-token KL for Gemma 4 E4B native Q8 artifacts."""

from __future__ import annotations

from pathlib import Path

import gemma4_e2b_q8_eval as refusal
import gemma4_e2b_q8_kl as engine


ROOT = Path(__file__).resolve().parents[1]
refusal.TOKENIZER_PATH = ROOT / "checkpoints" / "gemma4-e4b-it"
refusal.SCHEMA_FAMILY = "gemma4-e4b"
engine.RUN_DIR = ROOT / "runs" / "gemma4-e4b-q8" / "kl"
engine.RAW_LOGIT_SCHEMA = "gemma4-e4b-q8-first-token-raw-logits-v1"
engine.SCHEMA_FAMILY = "gemma4-e4b"


if __name__ == "__main__":
    engine.main()
