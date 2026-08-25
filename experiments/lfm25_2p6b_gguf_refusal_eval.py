#!/usr/bin/env python3
"""Evaluate an LM Studio GGUF with the frozen 2.6B refusal protocol."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from datasets import load_dataset
from transformers import AutoTokenizer

from experiments.lfm25_2p6b_residual_stream import (
    BAD_DATASET,
    BAD_REVISION,
    MAX_NEW_TOKENS,
    REFUSAL_MARKERS,
    base_path,
    normalized_response,
    render,
)
from experiments.lfm25_2p6b_gguf_comparator_eval import (
    LLAMA_CPP_BUILD,
    LLAMA_CPP_RUNTIME,
)
from heretic_nx.eval.gguf_runtime import native_completion
from heretic_nx.hashing import canonical_json, sha256_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="pinned GGUF artifact")
    parser.add_argument("--endpoint", default="http://127.0.0.1:1235")
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "runs" / "lfm25-2p6b-gguf-eval" / "report.json",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(base_path())
    rows = load_dataset(BAD_DATASET, revision=BAD_REVISION, split="test")
    rendered = render(
        tokenizer,
        [str(row["text"]) for row in rows],
        close_think=True,
    )
    prompts = [
        tokenizer.encode(prompt, add_special_tokens=False) for prompt in rendered
    ]
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        responses = list(
            pool.map(
                lambda prompt: native_completion(
                    args.endpoint,
                    prompt,
                    max_tokens=MAX_NEW_TOKENS,
                ),
                prompts,
            )
        )
    hits = [
        int(any(marker in normalized_response(text) for marker in REFUSAL_MARKERS))
        for text in responses
    ]
    report = {
        "schema_version": "lfm25-2p6b-gguf-refusal-native-v2",
        "model": args.model,
        "endpoint": args.endpoint,
        "runtime": {
            "engine": "llama.cpp",
            "distribution": LLAMA_CPP_RUNTIME,
            "build": LLAMA_CPP_BUILD,
            "api": "native /completion",
        },
        "dataset": {"id": BAD_DATASET, "revision": BAD_REVISION},
        "protocol": {
            "close_think": True,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": -1,
            "prompt_tokens_sha256": sha256_json(prompts),
            "bos_policy": "pre-tokenized prompt with exactly one explicit BOS",
            "refusal_markers": REFUSAL_MARKERS,
        },
        "count": len(responses),
        "refusal_markers": sum(hits),
        "refusals_per_100": 100 * sum(hits) / len(hits),
        "marker_hits": hits,
        "response_sha256": sha256_json(responses),
        "seconds": time.time() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_bytes(canonical_json(report) + b"\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
