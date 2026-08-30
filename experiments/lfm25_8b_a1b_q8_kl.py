#!/usr/bin/env python3
"""Exact full-vocabulary first-token KL for direct-Q8 LFM candidates."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from typing import Any
from urllib.request import Request, urlopen

from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer

from experiments.lfm25_2p6b_residual_stream import (
    GOOD_DATASET,
    GOOD_REVISION,
    render,
)
from heretic_nx.eval.kl_integrity import (
    first_token_kl,
    load_completed_log_probabilities,
    require_distinct_artifacts,
    require_matching_prompt_set,
    require_matching_runtime_protocol,
)
from heretic_nx.eval.gguf_runtime import (
    attest_native_model,
    require_native_model_identity,
)
from heretic_nx.hashing import canonical_json, sha256_json


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = ROOT / "checkpoints" / "lfm25-8b-a1b"
RUN_DIR = ROOT / "runs" / "lfm25-8b-a1b-q8-direct" / "kl"
VOCAB_SIZE = 128_000
BATCH_SIZE = 4
ROW_COUNT = 104
LOG_PROB_SCHEMA = "lfm25-8b-a1b-q8-first-token-logprobs-v1"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json(value) + b"\n")
    temporary.replace(path)


def log_probs(endpoint: str, tokens: list[int]) -> np.ndarray:
    payload = {
        "prompt": tokens,
        "n_predict": 1,
        "temperature": -1,
        "n_probs": VOCAB_SIZE,
        "stream": False,
    }
    request = Request(
        endpoint.rstrip("/") + "/completion",
        data=canonical_json(payload),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=300) as response:
        result = json.loads(response.read())
    if "error" in result:
        raise RuntimeError(str(result["error"]))
    rows = result["completion_probabilities"][0]["top_logprobs"]
    if len(rows) != VOCAB_SIZE:
        raise RuntimeError(f"llama.cpp returned {len(rows)} of {VOCAB_SIZE} logits")
    values = np.empty(VOCAB_SIZE, dtype=np.float32)
    seen = np.zeros(VOCAB_SIZE, dtype=np.bool_)
    for row in rows:
        token_id = int(row["id"])
        if not 0 <= token_id < VOCAB_SIZE or seen[token_id]:
            raise RuntimeError(f"invalid or duplicate token id: {token_id}")
        values[token_id] = float(row["logprob"])
        seen[token_id] = True
    if not bool(seen.all()) or not np.isfinite(values).all():
        raise RuntimeError("full-vocabulary log probabilities are incomplete")
    mass = float(np.exp(values.astype(np.float64)).sum())
    if abs(mass - 1.0) > 1e-3:
        raise RuntimeError(f"probability mass is not normalized: {mass}")
    return values


def prompts() -> tuple[list[list[int]], str]:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    rows = load_dataset(GOOD_DATASET, revision=GOOD_REVISION, split="test")
    rendered = render(
        tokenizer,
        [str(rows[index]["text"]) for index in range(104)],
        close_think=True,
    )
    token_rows = [
        tokenizer.encode(value, add_special_tokens=False) for value in rendered
    ]
    return token_rows, sha256_json(token_rows)


def collect(args: argparse.Namespace) -> None:
    token_rows, prompts_sha256 = prompts()
    runtime_model = attest_native_model(
        args.endpoint,
        args.artifact,
        expected_model=args.model,
    )
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    array_path = RUN_DIR / f"{args.label}.npy"
    progress_path = RUN_DIR / f"{args.label}.progress.json"
    expected = {
        "schema_version": LOG_PROB_SCHEMA,
        "label": args.label,
        "model": runtime_model["model_alias"],
        "prompt_tokens_sha256": prompts_sha256,
        "vocab_size": VOCAB_SIZE,
        "count": len(token_rows),
        "artifact_sha256": runtime_model["artifact_sha256"],
        "runtime_model": runtime_model,
    }
    progress = {**expected, "completed": 0, "seconds": 0.0}
    if progress_path.is_file():
        loaded = json.loads(progress_path.read_text(encoding="utf-8"))
        if not all(loaded.get(key) == value for key, value in expected.items()):
            raise RuntimeError(f"stale KL checkpoint: {progress_path}")
        progress = loaded
        matrix = np.lib.format.open_memmap(array_path, mode="r+")
        if matrix.shape != (len(token_rows), VOCAB_SIZE):
            raise RuntimeError(f"invalid KL matrix shape: {matrix.shape}")
    else:
        matrix = np.lib.format.open_memmap(
            array_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(token_rows), VOCAB_SIZE),
        )
        write_json(progress_path, progress)
    completed = int(progress["completed"])
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for start in range(completed, len(token_rows), BATCH_SIZE):
            require_native_model_identity(args.endpoint, runtime_model)
            batch = token_rows[start : start + BATCH_SIZE]
            started = time.time()
            produced = list(
                pool.map(lambda row: log_probs(args.endpoint, row), batch)
            )
            matrix[start : start + len(produced)] = np.stack(produced)
            matrix.flush()
            progress["seconds"] = float(progress["seconds"]) + (
                time.time() - started
            )
            progress["completed"] = start + len(produced)
            write_json(progress_path, progress)
            print(
                json.dumps(
                    {
                        "log_probs": args.label,
                        "completed": progress["completed"],
                        "total": len(token_rows),
                        "seconds": round(float(progress["seconds"]), 3),
                    }
                ),
                flush=True,
            )
    require_native_model_identity(
        args.endpoint,
        runtime_model,
        verify_artifact_hash=True,
    )
    print(
        json.dumps(
            {
                "label": args.label,
                "matrix": str(array_path),
                "shape": list(matrix.shape),
                "seconds": progress["seconds"],
            },
            indent=2,
        )
    )


def compare(args: argparse.Namespace) -> None:
    base_path = RUN_DIR / f"{args.base_label}.npy"
    candidate_path = RUN_DIR / f"{args.candidate_label}.npy"
    base_progress_path = RUN_DIR / f"{args.base_label}.progress.json"
    candidate_progress_path = RUN_DIR / f"{args.candidate_label}.progress.json"
    base, base_progress = load_completed_log_probabilities(
        base_path,
        base_progress_path,
        schema_version=LOG_PROB_SCHEMA,
        label=args.base_label,
        count=ROW_COUNT,
        vocab_size=VOCAB_SIZE,
    )
    candidate, candidate_progress = load_completed_log_probabilities(
        candidate_path,
        candidate_progress_path,
        schema_version=LOG_PROB_SCHEMA,
        label=args.candidate_label,
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
    values = []
    for row in range(base.shape[0]):
        values.append(first_token_kl(base[row], candidate[row]))
    result = {
        "schema_version": "lfm25-8b-a1b-q8-first-token-kl-v1",
        "base_label": args.base_label,
        "candidate_label": args.candidate_label,
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
        "count": len(values),
        "mean_first_token_kl": float(np.mean(values)),
        "maximum_first_token_kl": max(values),
        "median_first_token_kl": float(np.median(values)),
        "p95_first_token_kl": float(np.quantile(values, 0.95)),
        "per_row": values,
        "hard_cap": 0.03,
        "passed": float(np.mean(values)) <= 0.03,
    }
    report = RUN_DIR / f"{args.candidate_label}-vs-{args.base_label}.json"
    write_json(report, result)
    print(json.dumps({**result, "report": str(report)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--label", required=True)
    collect_parser.add_argument("--model", required=True)
    collect_parser.add_argument("--artifact", type=Path, required=True)
    collect_parser.add_argument("--endpoint", default="http://127.0.0.1:1236")
    collect_parser.add_argument("--parallel", type=int, default=4)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--base-label", required=True)
    compare_parser.add_argument("--candidate-label", required=True)
    args = parser.parse_args()
    if args.command == "collect":
        if args.parallel <= 0:
            raise ValueError("parallel must be positive")
        collect(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
