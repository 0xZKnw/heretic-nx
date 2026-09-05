#!/usr/bin/env python3
"""Prepare and sweep provenance-bound direct-Q8 Gemma 4 E4B edits."""

from __future__ import annotations

from pathlib import Path

import gemma4_e2b_q8_build as engine


ROOT = Path(__file__).resolve().parents[1]

engine.SOURCE = ROOT / "checkpoints" / "gemma4-e4b-it"
engine.SOURCE_WEIGHTS = engine.SOURCE / "model.safetensors"
engine.BASE_Q8 = (
    ROOT / "checkpoints" / "gemma4-e4b-gguf" / "gemma-4-E4B-it-Q8_0.gguf"
)
engine.OLD_RUN_DIR = ROOT / "runs" / "gemma4-e4b-residual-stream-prime"
engine.ACTIVATIONS = engine.OLD_RUN_DIR / "local-site-outputs.safetensors"
engine.DETECTORS = (
    engine.OLD_RUN_DIR / "teacher-delta-distilled-detectors.safetensors"
)
engine.DETECTOR_DIAGNOSTICS = (
    engine.OLD_RUN_DIR / "teacher-delta-distilled-diagnostics.json"
)
engine.RUN_DIR = ROOT / "runs" / "gemma4-e4b-q8"
engine.FACTORS = engine.RUN_DIR / "lambda100-factors.safetensors"
engine.PREPARATION = engine.RUN_DIR / "lambda100-preparation.json"
engine.MODEL_ID = "google/gemma-4-E4B-it"
engine.MODEL_REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"
engine.EXPECTED_STREAM_DIM = 2560
engine.EXPECTED_LAYER_COUNT = 42
engine.SCHEMA_FAMILY = "gemma4-e4b"
engine.ARTIFACT_STEM = "gemma4-e4b-q8"
engine.EXPECTED_SELECTED = (
    "L23:attention_out",
    "L06:ffn_out",
    "L35:attention_out",
    "L07:ffn_out",
    "L33:ffn_out",
    "L12:ffn_out",
    "L25:attention_out",
)


if __name__ == "__main__":
    engine.main()
