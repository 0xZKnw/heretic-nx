#!/usr/bin/env python3
"""Collect input states for distilling the compact eight-site Q8 teacher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from datasets import load_dataset
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
import numpy as np
from safetensors.torch import save_file
import torch
from transformers import AutoTokenizer

from experiments.lfm25_2p6b_residual_stream import (
    BAD_DATASET,
    BAD_REVISION,
    GOOD_DATASET,
    GOOD_REVISION,
    render,
)
from experiments.lfm25_8b_a1b_mlx_sites import MODEL_PATH, TOKENIZER_PATH
from experiments.lfm25_8b_a1b_q8_build import RUN_DIR
from experiments.lfm25_8b_a1b_q8_eval import MARKER_RULE_SHA256
from experiments.lfm25_8b_a1b_q8_prime_build import (
    OPERATORS,
    RANKING_REPORT,
    load_verified_prime_merge,
)
from heretic_nx.data.research_splits import (
    assert_research_splits_disjoint,
    build_research_split,
    manifest_from_report,
    verify_manifest_texts,
)
from heretic_nx.hashing import (
    canonical_json,
    sha256_directory,
    sha256_file,
    sha256_json,
)


TEACHER_MERGE = RUN_DIR / "prime-ops8-b2.merge.json"
BASE_REPORT = RUN_DIR / "base-train-geometry.json"
TEACHER_REPORT = RUN_DIR / "prime-ops8-b2-train-geometry.json"
BASE_SELECTION_REPORT = RUN_DIR / "base-validation-search.json"
TEACHER_SELECTION_REPORT = RUN_DIR / "prime-ops8-b2-validation-search.json"
OUTPUT = RUN_DIR / "teacher-op-inputs.safetensors"
MANIFEST = RUN_DIR / "teacher-op-inputs.manifest.json"
SAFE_ARRAY = RUN_DIR / "teacher-op-inputs.safe.npy"
HARMFUL_ARRAY = RUN_DIR / "teacher-op-inputs.harmful.npy"
PROGRESS = RUN_DIR / "teacher-op-inputs.progress.json"
SAFE_COUNT = 1024
SAFE_POOL_COUNT = 4096
TARGET_POOL_COUNT = 400
PAIR_COUNT = 32
WIDTH = 2048
MAX_LENGTH = 512
SPLIT_SEED = 20260830


class CaptureLinear(nn.Module):
    def __init__(self, linear: nn.Module):
        super().__init__()
        self.linear = linear
        self.value: mx.array | None = None

    def __call__(self, value: mx.array) -> mx.array:
        self.value = value
        return self.linear(value)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value) + b"\n")


def selected_sites(report: dict[str, object]) -> list[dict[str, object]]:
    rows = list(report["candidate"]["selected"])
    if len(rows) != 8 or any(row["family"] not in {"gqa", "liv"} for row in rows):
        raise RuntimeError("the pinned teacher must contain eight operator sites")
    if len({str(row["site_id"]) for row in rows}) != len(rows):
        raise RuntimeError("the pinned teacher repeats a semantic site")
    return rows


def install_captures(model: object, sites: list[dict[str, object]]) -> list[CaptureLinear]:
    captures = []
    for row in sites:
        layer = model.model.layers[int(row["layer"])]
        operator = layer.self_attn if row["family"] == "gqa" else layer.conv
        capture = CaptureLinear(operator.out_proj)
        operator.out_proj = capture
        captures.append(capture)
    return captures


def captured_stack(captures: list[CaptureLinear], start: int | None) -> np.ndarray:
    values = []
    for capture in captures:
        if capture.value is None:
            raise RuntimeError("an operator input capture did not fire")
        value = capture.value[0]
        value = value[-1:] if start is None else value[start:]
        values.append(value.astype(mx.float32))
    mx.eval(*values)
    stacked = np.stack([np.asarray(value) for value in values], axis=1)
    if stacked.ndim != 3 or stacked.shape[1:] != (len(captures), WIDTH):
        raise RuntimeError(f"invalid captured operator inputs: {stacked.shape}")
    if not np.isfinite(stacked).all():
        raise RuntimeError("captured operator inputs contain non-finite values")
    return stacked.astype(np.float16, copy=False)


def encode(tokenizer: object, value: str) -> list[int]:
    tokens = tokenizer.encode(value, add_special_tokens=False)
    if len(tokens) > MAX_LENGTH:
        tokens = [tokens[0], *tokens[-(MAX_LENGTH - 1) :]]
    return [int(token) for token in tokens]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--safe-count", type=int, default=SAFE_COUNT)
    parser.add_argument("--safe-pool-count", type=int, default=SAFE_POOL_COUNT)
    parser.add_argument("--target-pool-count", type=int, default=TARGET_POOL_COUNT)
    parser.add_argument("--pair-count", type=int, default=PAIR_COUNT)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--base-report", type=Path, default=BASE_REPORT)
    parser.add_argument("--teacher-report", type=Path, default=TEACHER_REPORT)
    parser.add_argument(
        "--base-selection-report",
        type=Path,
        default=BASE_SELECTION_REPORT,
    )
    parser.add_argument(
        "--teacher-selection-report",
        type=Path,
        default=TEACHER_SELECTION_REPORT,
    )
    args = parser.parse_args()
    if not 1 <= args.safe_count <= SAFE_COUNT:
        raise ValueError(f"safe-count must be between 1 and {SAFE_COUNT}")
    if not 1 <= args.pair_count <= PAIR_COUNT:
        raise ValueError(f"pair-count must be between 1 and {PAIR_COUNT}")
    if args.safe_pool_count < args.safe_count:
        raise ValueError("safe-pool-count must be at least safe-count")
    if args.target_pool_count <= 0:
        raise ValueError("target-pool-count must be positive")
    required = (
        args.model,
        TEACHER_MERGE,
        args.base_report,
        args.teacher_report,
        args.base_selection_report,
        args.teacher_selection_report,
    )
    if not all(path.exists() for path in required):
        raise RuntimeError(
            "the model or leak-safe geometry/selection teacher evidence is missing"
        )

    teacher_merge, ranking = load_verified_prime_merge(
        TEACHER_MERGE,
        verify_output=True,
    )
    sites = selected_sites(teacher_merge)
    source_sha256 = str(teacher_merge["source"]["sha256"])
    teacher_sha256 = str(teacher_merge["output"]["sha256"])
    model_sha256 = sha256_directory(args.model)
    geometry_provenance = ranking["geometry_provenance"]
    if (
        Path(str(geometry_provenance["model"])).resolve()
        != args.model.resolve()
        or geometry_provenance["model_sha256"] != model_sha256
    ):
        raise RuntimeError("teacher-input model differs from the geometry model")
    operator_sha256 = sha256_file(OPERATORS)
    ranking_sha256 = sha256_file(RANKING_REPORT)
    selected_sites_sha256 = sha256_json(sites)
    base = json.loads(args.base_report.read_text(encoding="utf-8"))
    teacher = json.loads(args.teacher_report.read_text(encoding="utf-8"))
    base_selection = json.loads(
        args.base_selection_report.read_text(encoding="utf-8")
    )
    teacher_selection = json.loads(
        args.teacher_selection_report.read_text(encoding="utf-8")
    )
    target_rows = load_dataset(
        BAD_DATASET,
        revision=BAD_REVISION,
        split=f"train[:{args.target_pool_count}]",
    )
    target_pool = [str(row["text"]) for row in target_rows]
    expected_geometry_manifest = build_research_split(
        target_pool,
        purpose="geometry",
        dataset_id=BAD_DATASET,
        revision=BAD_REVISION,
        source_split="train",
        seed=args.split_seed,
    )
    expected_selection_manifest = build_research_split(
        target_pool,
        purpose="selection",
        dataset_id=BAD_DATASET,
        revision=BAD_REVISION,
        source_split="train",
        seed=args.split_seed,
    )
    assert_research_splits_disjoint(
        expected_geometry_manifest,
        expected_selection_manifest,
    )
    base_geometry_manifest = manifest_from_report(
        base,
        expected_purpose="geometry",
        expected_artifact_sha256=source_sha256,
        expected_schema_version="lfm25-8b-a1b-q8-refusal-v2",
        expected_protocol_schema_version="lfm25-8b-a1b-q8-refusal-partial-v2",
        expected_marker_rule_sha256=MARKER_RULE_SHA256,
        expected_full_manifest_sha256=expected_geometry_manifest.sha256,
    )
    teacher_geometry_manifest = manifest_from_report(
        teacher,
        expected_purpose="geometry",
        expected_artifact_sha256=teacher_sha256,
        expected_schema_version="lfm25-8b-a1b-q8-refusal-v2",
        expected_protocol_schema_version="lfm25-8b-a1b-q8-refusal-partial-v2",
        expected_marker_rule_sha256=MARKER_RULE_SHA256,
        expected_full_manifest_sha256=expected_geometry_manifest.sha256,
    )
    base_selection_manifest = manifest_from_report(
        base_selection,
        expected_purpose="selection",
        expected_artifact_sha256=source_sha256,
        expected_schema_version="lfm25-8b-a1b-q8-refusal-v2",
        expected_protocol_schema_version="lfm25-8b-a1b-q8-refusal-partial-v2",
        expected_marker_rule_sha256=MARKER_RULE_SHA256,
        expected_full_manifest_sha256=expected_selection_manifest.sha256,
    )
    teacher_selection_manifest = manifest_from_report(
        teacher_selection,
        expected_purpose="selection",
        expected_artifact_sha256=teacher_sha256,
        expected_schema_version="lfm25-8b-a1b-q8-refusal-v2",
        expected_protocol_schema_version="lfm25-8b-a1b-q8-refusal-partial-v2",
        expected_marker_rule_sha256=MARKER_RULE_SHA256,
        expected_full_manifest_sha256=expected_selection_manifest.sha256,
    )
    if base_geometry_manifest.sha256 != teacher_geometry_manifest.sha256:
        raise RuntimeError("base and teacher geometry reports use different rows")
    if base_selection_manifest.sha256 != teacher_selection_manifest.sha256:
        raise RuntimeError("base and teacher selection reports use different rows")
    if (
        base_geometry_manifest.dataset_id != BAD_DATASET
        or base_geometry_manifest.revision != BAD_REVISION
        or base_geometry_manifest.seed != args.split_seed
        or base_selection_manifest.dataset_id != BAD_DATASET
        or base_selection_manifest.revision != BAD_REVISION
        or base_selection_manifest.seed != args.split_seed
        or base_geometry_manifest.pool_size != base_selection_manifest.pool_size
    ):
        raise RuntimeError("teacher reports do not use the pinned research pool")
    assert_research_splits_disjoint(
        base_geometry_manifest,
        base_selection_manifest,
    )
    if (
        base["protocol"].get("prompt_tokens_sha256")
        != teacher["protocol"].get("prompt_tokens_sha256")
        or base_selection["protocol"].get("prompt_tokens_sha256")
        != teacher_selection["protocol"].get("prompt_tokens_sha256")
    ):
        raise RuntimeError("paired base and teacher reports use different prompts")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    safe_rows = load_dataset(
        GOOD_DATASET,
        revision=GOOD_REVISION,
        split=f"train[:{args.safe_pool_count}]",
    )
    safe_pool = [str(row["text"]) for row in safe_rows]
    safe_manifest = build_research_split(
        safe_pool,
        purpose="geometry",
        dataset_id=GOOD_DATASET,
        revision=GOOD_REVISION,
        source_split="train",
        seed=args.split_seed,
        count=args.safe_count,
    )
    safe_texts = verify_manifest_texts(safe_manifest, safe_pool)
    safe_rendered = render(
        tokenizer,
        safe_texts,
        close_think=True,
    )
    safe_tokens = [encode(tokenizer, value) for value in safe_rendered]

    if base_geometry_manifest.sha256 != expected_geometry_manifest.sha256:
        raise RuntimeError("geometry report does not cover the full frozen split")
    if base_selection_manifest.sha256 != expected_selection_manifest.sha256:
        raise RuntimeError("selection report does not cover the full frozen split")
    target_texts = verify_manifest_texts(base_geometry_manifest, target_pool)
    target_rendered = render(
        tokenizer,
        target_texts,
        close_think=True,
    )
    eligible = [
        index
        for index, (base_hit, teacher_hit) in enumerate(
            zip(base["marker_hits"], teacher["marker_hits"], strict=True)
        )
        if int(base_hit) == 1 and int(teacher_hit) == 0
    ]
    if len(eligible) < args.pair_count:
        raise RuntimeError("too few base-refusal/teacher-success trajectories")
    # Spread the limited trajectory budget across the frozen geometry split.
    positions = np.linspace(0, len(eligible) - 1, args.pair_count).round().astype(int)
    pair_positions = [eligible[int(position)] for position in positions]
    if len(pair_positions) != len(set(pair_positions)):
        raise RuntimeError("trajectory sampling produced duplicate positions")
    pair_source_indices = [
        base_geometry_manifest.rows[position].source_index
        for position in pair_positions
    ]
    harmful_rows = []
    response_starts = []
    for position in pair_positions:
        prompt = encode(tokenizer, target_rendered[position])
        response = tokenizer.encode(
            str(base["responses"][position]), add_special_tokens=False
        )
        combined = prompt + [int(token) for token in response]
        if len(combined) > MAX_LENGTH:
            response = response[: max(MAX_LENGTH - len(prompt), 1)]
            combined = prompt + [int(token) for token in response]
        harmful_rows.append(combined)
        response_starts.append(len(prompt))
    harmful_tokens = sum(len(row) - start for row, start in zip(harmful_rows, response_starts))

    expected = {
        "schema_version": "lfm25-8b-a1b-teacher-op-inputs-v3",
        "model": str(args.model.resolve()),
        "model_sha256": model_sha256,
        "safe_count": args.safe_count,
        "pair_count": args.pair_count,
        "pair_positions": pair_positions,
        "pair_source_indices": pair_source_indices,
        "site_ids": [str(row["site_id"]) for row in sites],
        "safe_split_manifest": safe_manifest.to_dict(),
        "safe_split_manifest_sha256": safe_manifest.sha256,
        "target_geometry_manifest": base_geometry_manifest.to_dict(),
        "target_geometry_manifest_sha256": base_geometry_manifest.sha256,
        "target_selection_manifest_sha256": base_selection_manifest.sha256,
        "base_geometry_report_sha256": sha256_file(args.base_report),
        "teacher_geometry_report_sha256": sha256_file(args.teacher_report),
        "base_selection_report_sha256": sha256_file(args.base_selection_report),
        "teacher_selection_report_sha256": sha256_file(
            args.teacher_selection_report
        ),
        "teacher_merge_sha256": sha256_file(TEACHER_MERGE),
        "ranking_report_sha256": ranking_sha256,
        "operator_artifact_sha256": operator_sha256,
        "geometry_provenance_sha256": sha256_json(geometry_provenance),
        "selected_sites_sha256": selected_sites_sha256,
        "source_artifact_sha256": source_sha256,
        "teacher_artifact_sha256": teacher_sha256,
        "safe_tokens_sha256": sha256_json(safe_tokens),
        "harmful_tokens_sha256": sha256_json(harmful_rows),
        "harmful_response_tokens": harmful_tokens,
    }
    progress = {
        **expected,
        "phase": "safe",
        "completed": 0,
        "offset": 0,
        "seconds": 0.0,
        "complete": False,
        "output_sha256": None,
    }
    mode = "w+"
    if PROGRESS.is_file():
        loaded = json.loads(PROGRESS.read_text(encoding="utf-8"))
        if not all(loaded.get(key) == value for key, value in expected.items()):
            raise RuntimeError(f"stale teacher-input checkpoint: {PROGRESS}")
        phase = loaded.get("phase")
        completed = loaded.get("completed")
        offset = loaded.get("offset")
        completed_limit = args.safe_count if phase == "safe" else args.pair_count
        if (
            phase not in {"safe", "harmful"}
            or isinstance(completed, bool)
            or not isinstance(completed, int)
            or not 0 <= completed <= completed_limit
            or isinstance(offset, bool)
            or not isinstance(offset, int)
            or not 0 <= offset <= harmful_tokens
            or not isinstance(loaded.get("complete"), bool)
        ):
            raise RuntimeError(f"invalid teacher-input checkpoint: {PROGRESS}")
        if loaded["complete"]:
            if not MANIFEST.is_file() or not OUTPUT.is_file():
                raise RuntimeError(
                    f"completed teacher-input artifacts are missing: {PROGRESS}"
                )
            document = json.loads(MANIFEST.read_text(encoding="utf-8"))
            document_output_sha256 = document.pop("output_sha256", None)
            document_manifest_sha256 = document.pop("manifest_sha256", None)
            if (
                phase != "harmful"
                or completed != args.pair_count
                or offset != harmful_tokens
                or document != expected
                or document_manifest_sha256 != sha256_json(expected)
                or document_output_sha256 != sha256_file(OUTPUT)
                or loaded.get("output_sha256") != document_output_sha256
            ):
                raise RuntimeError(
                    f"corrupt completed teacher-input checkpoint: {PROGRESS}"
                )
            print(
                json.dumps(
                    {
                        "output": str(OUTPUT),
                        "output_sha256": document_output_sha256,
                        "reused": True,
                    },
                    indent=2,
                ),
                flush=True,
            )
            return
        if loaded.get("output_sha256") is not None:
            raise RuntimeError(
                f"partial teacher-input checkpoint claims an output: {PROGRESS}"
            )
        progress = loaded
        mode = "r+"
    safe = np.lib.format.open_memmap(
        SAFE_ARRAY,
        mode=mode,
        dtype=np.float16,
        shape=(args.safe_count, len(sites), WIDTH),
    )
    harmful = np.lib.format.open_memmap(
        HARMFUL_ARRAY,
        mode=mode,
        dtype=np.float16,
        shape=(harmful_tokens, len(sites), WIDTH),
    )

    print(json.dumps({"load": str(args.model), "sites": len(sites)}), flush=True)
    model, _ = load(args.model, lazy=True)
    captures = install_captures(model, sites)
    if progress["phase"] == "safe":
        for index in range(int(progress["completed"]), len(safe_tokens)):
            started = time.time()
            inputs = mx.array([safe_tokens[index]], dtype=mx.int32)
            output = model.model(inputs)
            mx.eval(output)
            safe[index] = captured_stack(captures, None)[0]
            safe.flush()
            progress.update(
                {
                    "completed": index + 1,
                    "seconds": float(progress["seconds"]) + time.time() - started,
                }
            )
            write_json(PROGRESS, progress)
            if (index + 1) % 64 == 0 or index + 1 == len(safe_tokens):
                print(
                    json.dumps(
                        {
                            "collect": "safe",
                            "completed": index + 1,
                            "total": len(safe_tokens),
                            "seconds": round(float(progress["seconds"]), 3),
                        }
                    ),
                    flush=True,
                )
        progress.update({"phase": "harmful", "completed": 0, "offset": 0})
        write_json(PROGRESS, progress)

    offset = int(progress["offset"])
    for index in range(int(progress["completed"]), len(harmful_rows)):
        started = time.time()
        inputs = mx.array([harmful_rows[index]], dtype=mx.int32)
        output = model.model(inputs)
        mx.eval(output)
        values = captured_stack(captures, response_starts[index])
        stop = offset + len(values)
        harmful[offset:stop] = values
        harmful.flush()
        offset = stop
        progress.update(
            {
                "completed": index + 1,
                "offset": offset,
                "seconds": float(progress["seconds"]) + time.time() - started,
            }
        )
        write_json(PROGRESS, progress)
        print(
            json.dumps(
                {
                    "collect": "harmful",
                    "completed": index + 1,
                    "total": len(harmful_rows),
                    "tokens": offset,
                    "seconds": round(float(progress["seconds"]), 3),
                }
            ),
            flush=True,
        )
    if offset != harmful_tokens:
        raise RuntimeError(f"harmful token count mismatch: {offset} != {harmful_tokens}")
    save_file(
        {
            "safe": torch.from_numpy(np.asarray(safe).copy()),
            "harmful": torch.from_numpy(np.asarray(harmful).copy()),
        },
        OUTPUT,
        metadata={
            "manifest_sha256": sha256_json(expected),
            "teacher_merge_sha256": str(expected["teacher_merge_sha256"]),
            "ranking_report_sha256": ranking_sha256,
            "operators_sha256": operator_sha256,
            "source_artifact_sha256": source_sha256,
            "teacher_artifact_sha256": teacher_sha256,
            "selected_sites_sha256": selected_sites_sha256,
        },
    )
    output_sha256 = sha256_file(OUTPUT)
    write_json(
        MANIFEST,
        {
            **expected,
            "manifest_sha256": sha256_json(expected),
            "output_sha256": output_sha256,
        },
    )
    progress.update(
        {
            "phase": "harmful",
            "completed": args.pair_count,
            "offset": harmful_tokens,
            "complete": True,
            "output_sha256": output_sha256,
        }
    )
    write_json(PROGRESS, progress)
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "output_sha256": output_sha256,
                "safe_shape": list(safe.shape),
                "harmful_shape": list(harmful.shape),
                "seconds": progress["seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
