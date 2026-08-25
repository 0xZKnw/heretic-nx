#!/usr/bin/env python3
"""Validate the released PRIME Q8 GGUF on the full XSTest slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from experiments.lfm25_2p6b_gguf_comparator_eval import (
    LLAMA_CPP_BUILD,
    LLAMA_CPP_RUNTIME,
    resume_map,
    write_json,
)
from experiments.lfm25_2p6b_residual_stream import (
    MAX_NEW_TOKENS,
    REFUSAL_MARKERS,
    base_path,
    normalized_response,
    render,
)
from experiments.lfm25_xstest_retest import XSTEST_ID, XSTEST_REVISION
from heretic_nx.eval.gguf_runtime import native_completion
from heretic_nx.hashing import sha256_file, sha256_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = (
    ROOT.parent
    / "outputs"
    / "LFM2.5-2.6B-Heretic-NX-Residual-Stream-PRIME-v8"
    / "LFM2.5-2.6B-Heretic-NX-PRIME-Q8_0.gguf"
)
DEFAULT_RUN_DIR = ROOT / "runs" / "lfm25-2p6b-prime-q8"
PINNED_ARTIFACT_SHA256 = (
    "fc038207dc27fa279291c08a0228b4a37f2c38137e530a8a5802804bc1a6c1b7"
)


def agreement(left: list[int], right: list[int]) -> float:
    if len(left) != len(right) or not left:
        raise RuntimeError("marker-hit vectors must have equal nonzero length")
    return sum(int(a == b) for a, b in zip(left, right)) / len(left)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:1235")
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--expected-sha256",
        default=PINNED_ARTIFACT_SHA256,
    )
    parser.add_argument("--format-label", default="GGUF Q8_0")
    parser.add_argument(
        "--quantized",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--partial-name", default="xstest-b10621.partial.json")
    parser.add_argument("--report-name", default="xstest-b10621.json")
    args = parser.parse_args()
    args.artifact = args.artifact.resolve()
    args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.parallel <= 0:
        raise ValueError("parallel must be positive")
    for name in (args.partial_name, args.report_name):
        if Path(name).name != name or not name.endswith(".json"):
            raise ValueError(f"report names must be local JSON filenames: {name}")
    if not args.artifact.is_file():
        raise RuntimeError(f"missing PRIME Q8 GGUF artifact: {args.artifact}")
    artifact_hash = sha256_file(args.artifact)
    if artifact_hash != args.expected_sha256:
        raise RuntimeError(
            f"PRIME Q8 GGUF hash mismatch: {artifact_hash} != "
            f"{args.expected_sha256}"
        )

    dataset = load_dataset(
        XSTEST_ID,
        revision=XSTEST_REVISION,
        split="prompts",
    )
    rows = [dict(row) for row in dataset]
    row_payload = [
        {"id": row["id"], "type": row["type"], "prompt": row["prompt"]}
        for row in rows
    ]
    rows_hash = sha256_json(row_payload)
    unsafe = [str(row["type"]).startswith("contrast_") for row in rows]
    tokenizer = AutoTokenizer.from_pretrained(base_path())
    rendered = render(
        tokenizer,
        [str(row["prompt"]) for row in rows],
        close_think=True,
    )
    prompt_tokens = [
        tokenizer.encode(value, add_special_tokens=False) for value in rendered
    ]
    prompt_tokens_hash = sha256_json(prompt_tokens)
    expected = {
        "schema_version": "lfm25-2p6b-prime-q8-xstest-partial-v1",
        "artifact_sha256": artifact_hash,
        "rows_sha256": rows_hash,
        "prompt_tokens_sha256": prompt_tokens_hash,
        "max_new_tokens": MAX_NEW_TOKENS,
        "close_think": True,
        "temperature": -1,
        "runtime_build": LLAMA_CPP_BUILD,
    }
    responses, seconds = resume_map(
        prompts=prompt_tokens,
        worker=lambda tokens: native_completion(
            args.endpoint,
            tokens,
            max_tokens=MAX_NEW_TOKENS,
        ),
        partial_path=args.run_dir / args.partial_name,
        expected=expected,
        parallel=args.parallel,
        encode=str,
    )
    hits = [
        int(any(marker in normalized_response(text) for marker in REFUSAL_MARKERS))
        for text in responses
    ]
    safe_hits = [hit for hit, is_unsafe in zip(hits, unsafe) if not is_unsafe]
    unsafe_hits = [hit for hit, is_unsafe in zip(hits, unsafe) if is_unsafe]

    native_report_path = (
        ROOT / "runs" / "lfm25-2p6b-eval-prime-v8" / "xstest.json"
    )
    native_report = json.loads(native_report_path.read_text(encoding="utf-8"))
    native = native_report["results"]["residual_stream"]
    native_safe = [int(value) for value in native["safe_marker_hits"]]
    native_unsafe = [int(value) for value in native["unsafe_marker_hits"]]
    report = {
        "schema_version": "lfm25-2p6b-prime-q8-xstest-v1",
        "artifact": {
            "filename": args.artifact.name,
            "sha256": artifact_hash,
            "size_bytes": args.artifact.stat().st_size,
            "format": args.format_label,
            "quantized": args.quantized,
        },
        "runtime": {
            "engine": "llama.cpp",
            "distribution": LLAMA_CPP_RUNTIME,
            "build": LLAMA_CPP_BUILD,
            "endpoint": args.endpoint,
            "api": "native /completion",
        },
        "native_reference": {
            "report": str(native_report_path.relative_to(ROOT)),
            "report_sha256": sha256_file(native_report_path),
            "format": "Transformers BF16",
        },
        "dataset": {
            "id": XSTEST_ID,
            "revision": XSTEST_REVISION,
            "rows_sha256": rows_hash,
        },
        "protocol": {
            "prompt_tokens_sha256": prompt_tokens_hash,
            "close_think": True,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": -1,
            "refusal_markers": REFUSAL_MARKERS,
            "metric": "lexical refusal-marker proxy",
        },
        "results": {
            "all_count": len(hits),
            "all_refusal_markers": sum(hits),
            "safe_count": len(safe_hits),
            "safe_refusal_markers": sum(safe_hits),
            "unsafe_count": len(unsafe_hits),
            "unsafe_refusal_markers": sum(unsafe_hits),
            "safe_marker_hits": safe_hits,
            "unsafe_marker_hits": unsafe_hits,
            "response_sha256": sha256_json(responses),
            "seconds": seconds,
            "responses_per_second": len(responses) / max(seconds, 1e-9),
            "safe_marker_agreement_with_native": agreement(
                safe_hits,
                native_safe,
            ),
            "unsafe_marker_agreement_with_native": agreement(
                unsafe_hits,
                native_unsafe,
            ),
            "native_all_refusal_markers": int(native["all_refusal_markers"]),
            "native_safe_refusal_markers": int(native["safe_refusal_markers"]),
            "native_unsafe_refusal_markers": int(
                native["unsafe_refusal_markers"]
            ),
        },
        "interpretation_guard": (
            "Lexical refusal markers are not semantic task-success or safety "
            "judgments. This report validates the quantized runtime artifact."
        ),
    }
    write_json(args.run_dir / args.report_name, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
