#!/usr/bin/env python3
"""Qwen3.5-9B Q8 refusal evaluation with the frozen 104-row protocol."""
from pathlib import Path

import gemma4_e2b_q8_eval as engine
from heretic_nx.eval.gguf_runtime import NativeRuntimeClient


class IndependentPromptClient(NativeRuntimeClient):
    """Disable prompt reuse explicitly for Qwen's recurrent blocks."""

    def completion(self, prompt_tokens, *, max_tokens):
        result = self._http.request_json(
            "/completion",
            payload={"prompt": prompt_tokens, "n_predict": max_tokens,
                     "temperature": -1, "stream": False, "cache_prompt": False},
            timeout=300, failure="Qwen independent native generation failed",
        )
        return self._content(result, failure="Qwen native generation failed")

ROOT = Path(__file__).resolve().parents[1]
engine.TOKENIZER_PATH = ROOT / "checkpoints/qwen35-9b-tokenizer"
engine.RUN_DIR = ROOT / "runs/qwen35-9b-q8/refusal"
engine.SCHEMA_FAMILY = "qwen35-9b"
engine.NativeRuntimeClient = IndependentPromptClient
_write_json = engine.write_json


def write_json(path, value):
    if "protocol" in value:
        value = {**value, "protocol": {**value["protocol"],
            "enable_thinking": False, "cache_prompt": False,
            "system_prompt": engine.SYSTEM_PROMPT}}
    _write_json(path, value)


engine.write_json = write_json

if __name__ == "__main__":
    engine.main()
