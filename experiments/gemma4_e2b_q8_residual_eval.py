#!/usr/bin/env python3
"""Fast native-Q8 screen over the union of prior refusal-marker rows."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from datasets import load_dataset
from transformers import AutoTokenizer

from heretic_nx.eval.gguf_runtime import (
    NativeRuntimeClient,
    attest_native_model,
    require_native_model_identity,
)
from heretic_nx.hashing import sha256_json

import gemma4_e2b_q8_eval as full


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--source-reports", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:1236")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=full.MAX_NEW_TOKENS)
    args = parser.parse_args()

    indices: set[int] = set()
    sources = []
    for raw in args.source_reports.split(","):
        path = Path(raw).expanduser().resolve(strict=True)
        report = json.loads(path.read_text(encoding="utf-8"))
        hits = report.get("marker_hits")
        if not isinstance(hits, list):
            raise RuntimeError(f"refusal report has no marker hits: {path}")
        row_indices = report.get("row_indices_zero_based")
        if row_indices is None and len(hits) == full.ROW_COUNT:
            row_indices = list(range(full.ROW_COUNT))
        if not isinstance(row_indices, list) or len(row_indices) != len(hits):
            raise RuntimeError(f"refusal report has no valid row mapping: {path}")
        indices.update(
            int(row_index)
            for row_index, hit in zip(row_indices, hits)
            if int(hit) == 1
        )
        sources.append({"path": str(path), "evidence_sha256": report["evidence_sha256"]})
    selected = sorted(indices)
    if not selected:
        raise RuntimeError("source reports contain no refusal-marker rows")

    runtime = attest_native_model(args.endpoint, args.artifact, expected_model=args.model)
    tokenizer = AutoTokenizer.from_pretrained(full.TOKENIZER_PATH, local_files_only=True)
    rows = load_dataset(full.BAD_DATASET, revision=full.BAD_REVISION, split="test")
    prompts = [str(rows[index]["text"]) for index in selected]
    prompt_tokens = [
        tokenizer.encode(value, add_special_tokens=False)
        for value in full.render(tokenizer, prompts)
    ]
    client = NativeRuntimeClient(args.endpoint)
    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            responses = list(
                pool.map(
                    lambda tokens: client.completion(
                        tokens, max_tokens=args.max_new_tokens
                    ),
                    prompt_tokens,
                )
            )
    finally:
        client.close()
    require_native_model_identity(args.endpoint, runtime, verify_artifact_hash=True)
    hits = [
        int(
            any(
                marker in full.normalized_response(response)
                for marker in full.REFUSAL_MARKERS
            )
        )
        for response in responses
    ]
    report = {
        "schema_version": "gemma4-e2b-q8-residual-screen-v1",
        "label": args.label,
        "runtime_model": runtime,
        "source_reports": sources,
        "row_indices_zero_based": selected,
        "row_indices_one_based": [index + 1 for index in selected],
        "count": len(selected),
        "refusal_markers": sum(hits),
        "marker_hits": hits,
        "prompt_tokens_sha256": sha256_json(prompt_tokens),
        "response_sha256": sha256_json(responses),
        "evidence_sha256": sha256_json({"indices": selected, "hits": hits, "responses": responses}),
        "seconds": time.perf_counter() - started,
        "responses": responses,
    }
    report_path = full.RUN_DIR / f"{args.label}.residual.json"
    full.write_json(report_path, report)
    print(
        json.dumps(
            {
                "label": args.label,
                "count": len(selected),
                "refusal_markers": sum(hits),
                "report": str(report_path),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
