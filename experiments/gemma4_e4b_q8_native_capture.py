#!/usr/bin/env python3
"""Capture all pinned benign-test inputs from the native E4B Q8 runtime."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile

from heretic_nx.hashing import canonical_json, sha256_file, sha256_json

import gemma4_e4b_q8_build as parent
import gemma4_e4b_q8_kl as kl


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "gemma4-e4b-q8"
OUTPUT = RUN_DIR / "q8-benign104-inputs"
MANIFEST = RUN_DIR / "q8-benign104-inputs.json"
EXECUTABLE = ROOT / "build" / "llama.cpp-native" / "bin" / "llama_capture_weight_inputs"


def main() -> None:
    preparation = parent.engine._verified_preparation()
    token_rows, token_hash, tokenizer_identity = kl.engine.prompts()
    weights = [str(row["tensor_name"]) for row in preparation["selected"]]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as stream:
        for row in token_rows:
            stream.write(" ".join(str(token) for token in row) + "\n")
        stream.flush()
        completed = subprocess.run(
            [
                str(EXECUTABLE),
                str(parent.engine.BASE_Q8),
                stream.name,
                str(OUTPUT),
                ",".join(weights),
            ],
            check=True,
            text=True,
            capture_output=True,
        )
    # The collector can also write row*.logits.f32 controls; only projection
    # inputs belong in the site-input count and digest below.
    files = sorted(OUTPUT.glob("row*.site*.f32"))
    expected = len(token_rows) * len(weights)
    if len(files) != expected:
        raise RuntimeError(f"captured {len(files)} native inputs, expected {expected}")
    manifest = {
        "schema_version": "gemma4-e4b-q8-native-benign-inputs-v1",
        "model": {
            "path": str(parent.engine.BASE_Q8),
            "sha256": sha256_file(parent.engine.BASE_Q8),
        },
        "prompt_tokens_sha256": token_hash,
        "tokenizer": tokenizer_identity,
        "row_count": len(token_rows),
        "site_count": len(weights),
        "weights": weights,
        "files_sha256": sha256_json(
            [{"name": path.name, "sha256": sha256_file(path)} for path in files]
        ),
        "runtime_stdout": completed.stdout.strip(),
    }
    MANIFEST.write_bytes(canonical_json(manifest) + b"\n")
    print(json.dumps({**manifest, "manifest": str(MANIFEST)}, indent=2))


if __name__ == "__main__":
    main()
