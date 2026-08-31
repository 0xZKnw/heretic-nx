#!/usr/bin/env python3
"""Exact full-vocabulary first-token KL for Gemma 4 E2B native Q8 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer

from heretic_nx.eval.kl_integrity import (
    default_progress_path,
    first_token_kl,
    load_completed_raw_logits,
    require_distinct_artifacts,
    require_matching_prompt_set,
    require_matching_runtime_protocol,
)
from heretic_nx.eval.native_logits import (
    attest_tokenizer_assets,
    collect_native_raw_logits,
)
from heretic_nx.hashing import canonical_json, sha256_json

import gemma4_e2b_q8_eval as refusal


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "gemma4-e2b-q8" / "kl"
GOOD_DATASET = "mlabonne/harmless_alpaca"
GOOD_REVISION = "02c6a92cfcf11bb0c387334f8146d149d65b587f"
ROW_COUNT = 104
VOCAB_SIZE = 262_144
RAW_LOGIT_SCHEMA = "gemma4-e2b-q8-first-token-raw-logits-v1"
NATIVE_RUNTIME_DIR = ROOT / "build" / "llama.cpp-native" / "bin"
NATIVE_EXECUTABLE = NATIVE_RUNTIME_DIR / "llama_raw_logits"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json(value) + b"\n")
    temporary.replace(path)


def prompts() -> tuple[list[list[int]], str, dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(
        refusal.TOKENIZER_PATH, local_files_only=True
    )
    if len(tokenizer) != VOCAB_SIZE:
        raise RuntimeError(
            f"tokenizer vocabulary mismatch: {len(tokenizer)} != {VOCAB_SIZE}"
        )
    rows = load_dataset(GOOD_DATASET, revision=GOOD_REVISION, split="test")
    rendered = refusal.render(
        tokenizer,
        [str(rows[index]["text"]) for index in range(ROW_COUNT)],
    )
    token_rows = [
        tokenizer.encode(value, add_special_tokens=False) for value in rendered
    ]
    identity = attest_tokenizer_assets(
        refusal.TOKENIZER_PATH,
        vocab_size=len(tokenizer),
        tokenizer_class=f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
    )
    return token_rows, sha256_json(token_rows), identity


def collect(args: argparse.Namespace) -> None:
    token_rows, prompt_hash, tokenizer_identity = prompts()
    output = Path(args.output) if args.output else RUN_DIR / f"{args.label}.raw.bin"
    result = collect_native_raw_logits(
        token_rows=token_rows,
        tokenizer_identity=tokenizer_identity,
        model_path=args.artifact,
        output_path=output,
        progress_path=args.progress,
        schema_version=RAW_LOGIT_SCHEMA,
        label=args.label,
        model_alias=args.model,
        executable_path=args.executable,
        runtime_library_dirs=args.runtime_dir or [NATIVE_RUNTIME_DIR],
        context_size=args.context_size,
        batch_size=args.batch_size,
        ubatch_size=args.ubatch_size,
        threads=args.threads,
        gpu_layers=args.gpu_layers,
        timeout_seconds=args.timeout,
    )
    if result.progress["prompt_tokens_sha256"] != prompt_hash:
        raise RuntimeError("native collector prompt identity mismatch")
    print(
        json.dumps(
            {
                "label": args.label,
                "matrix": str(result.data_path),
                "progress": str(result.progress_path),
                "shape": [ROW_COUNT, VOCAB_SIZE],
                "seconds": result.progress["seconds"],
                "process_seconds": result.progress["process_seconds"],
                "reused": result.reused,
                "artifact_sha256": result.progress["artifact_sha256"],
                "data_sha256": result.progress["data_sha256"],
            },
            indent=2,
        ),
        flush=True,
    )


def normalized_log_probs(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    maximum = float(np.max(values))
    return values - (maximum + np.log(np.exp(values - maximum).sum()))


def compare(args: argparse.Namespace) -> None:
    base_path = Path(args.base)
    candidate_path = Path(args.candidate)
    base_progress_path = (
        Path(args.base_progress)
        if args.base_progress
        else default_progress_path(base_path)
    )
    candidate_progress_path = (
        Path(args.candidate_progress)
        if args.candidate_progress
        else default_progress_path(candidate_path)
    )
    base, base_progress = load_completed_raw_logits(
        base_path,
        base_progress_path,
        schema_version=RAW_LOGIT_SCHEMA,
        count=ROW_COUNT,
        vocab_size=VOCAB_SIZE,
    )
    candidate, candidate_progress = load_completed_raw_logits(
        candidate_path,
        candidate_progress_path,
        schema_version=RAW_LOGIT_SCHEMA,
        count=ROW_COUNT,
        vocab_size=VOCAB_SIZE,
    )
    require_matching_prompt_set(
        base_progress,
        candidate_progress,
        base_path=base_progress_path,
        candidate_path=candidate_progress_path,
    )
    require_distinct_artifacts(base_progress, candidate_progress)
    require_matching_runtime_protocol(base_progress, candidate_progress)
    values = [
        first_token_kl(
            normalized_log_probs(base[index]),
            normalized_log_probs(candidate[index]),
        )
        for index in range(ROW_COUNT)
    ]
    mean = float(np.mean(values))
    report = {
        "schema_version": "gemma4-e2b-q8-first-token-kl-raw-v1",
        "base": str(base_path),
        "candidate": str(candidate_path),
        "prompt_tokens_sha256": base_progress["prompt_tokens_sha256"],
        "base_artifact": {
            "model": base_progress["model"],
            "sha256": base_progress["artifact_sha256"],
            "runtime": base_progress["runtime_model"],
        },
        "candidate_artifact": {
            "model": candidate_progress["model"],
            "sha256": candidate_progress["artifact_sha256"],
            "runtime": candidate_progress["runtime_model"],
        },
        "count": ROW_COUNT,
        "vocab_size": VOCAB_SIZE,
        "mean_first_token_kl": mean,
        "maximum_first_token_kl": max(values),
        "median_first_token_kl": float(np.median(values)),
        "p95_first_token_kl": float(np.quantile(values, 0.95)),
        "per_row": values,
        "hard_cap": 0.05,
        "passed": mean <= 0.05,
    }
    report_path = Path(args.report)
    write_json(report_path, report)
    print(json.dumps({**report, "report": str(report_path)}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--label", required=True)
    collect_parser.add_argument("--model", required=True)
    collect_parser.add_argument("--artifact", type=Path, required=True)
    collect_parser.add_argument("--executable", type=Path, default=NATIVE_EXECUTABLE)
    collect_parser.add_argument("--runtime-dir", type=Path, action="append")
    collect_parser.add_argument("--output", type=Path)
    collect_parser.add_argument("--progress", type=Path)
    collect_parser.add_argument("--context-size", type=int)
    collect_parser.add_argument("--batch-size", type=int)
    collect_parser.add_argument("--ubatch-size", type=int)
    collect_parser.add_argument("--threads", type=int, default=4)
    collect_parser.add_argument("--gpu-layers", type=int, default=-1)
    collect_parser.add_argument("--timeout", type=float)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--base", required=True)
    compare_parser.add_argument("--candidate", required=True)
    compare_parser.add_argument("--report", required=True)
    compare_parser.add_argument("--base-progress")
    compare_parser.add_argument("--candidate-progress")
    args = parser.parse_args()
    if args.command == "collect":
        collect(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
