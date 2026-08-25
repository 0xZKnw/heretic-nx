#!/usr/bin/env python3
"""Evaluate an LM Studio GGUF with the frozen 2.6B refusal protocol."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

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
from heretic_nx.hashing import canonical_json, sha256_json


ROOT = Path(__file__).resolve().parents[1]


def completion(endpoint: str, model: str, prompt: str) -> str:
    payload = canonical_json(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": MAX_NEW_TOKENS,
            "temperature": 0,
            "stream": False,
        }
    )
    request = Request(
        endpoint.rstrip("/") + "/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=300) as response:
        result = json.loads(response.read())
    return str(result["choices"][0]["text"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:1234")
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "runs" / "lfm25-2p6b-gguf-eval" / "report.json",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(base_path())
    rows = load_dataset(BAD_DATASET, revision=BAD_REVISION, split="test")
    prompts = render(
        tokenizer,
        [str(row["text"]) for row in rows],
        close_think=True,
    )
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        responses = list(
            pool.map(
                lambda prompt: completion(args.endpoint, args.model, prompt),
                prompts,
            )
        )
    hits = [
        int(any(marker in normalized_response(text) for marker in REFUSAL_MARKERS))
        for text in responses
    ]
    report = {
        "schema_version": "lfm25-2p6b-gguf-refusal-v1",
        "model": args.model,
        "endpoint": args.endpoint,
        "dataset": {"id": BAD_DATASET, "revision": BAD_REVISION},
        "protocol": {
            "close_think": True,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": 0,
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
