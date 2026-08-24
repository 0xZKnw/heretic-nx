#!/usr/bin/env python3
"""Paired external evaluation of LFM2.5 editing-engine artifacts.

Every arm is evaluated with the same rendered prompts, greedy decoding, lexical
target proxy, and deterministic multiple-choice capability slice. The report
retains item-level binary observations so confidence intervals can be recomputed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import random

from datasets import load_dataset

from experiments.lfm25_residual_stream_capability import (
    DATASETS,
    LETTERS,
    NONINFERIORITY_MARGIN,
    SEED,
    evaluate as evaluate_capability,
    prompt_text,
    selected_rows,
    task_scores,
)
from experiments.lfm25_xstest_retest import (
    XSTEST_ID,
    XSTEST_REVISION,
    evaluate as evaluate_xstest,
)
from heretic_nx.eval.capability import paired_bootstrap_interval
from heretic_nx.hashing import canonical_json, sha256_file, sha256_json


STRONGREJECT_ID = "AlignmentResearch/StrongREJECT"
STRONGREJECT_REVISION = "9f4b3397bda985e1ac6dadb655e9ca3a17974144"
STRONGREJECT_SPLIT = "validation"


def expanded_capability_rows() -> list[dict]:
    """Extend the frozen 241-row slice deterministically without replacing it."""
    rows = selected_rows()
    seen = {row["id"] for row in rows}
    generator = random.Random(SEED + 1)

    arc_spec = DATASETS["arc_challenge"]
    arc = load_dataset(
        arc_spec["id"],
        arc_spec["config"],
        revision=arc_spec["revision"],
        split=arc_spec["split"],
    )
    arc_candidates = []
    for index, row in enumerate(arc):
        labels = list(row["choices"]["label"])
        if len(labels) == 4 and set(labels) == set(LETTERS) and f"arc:{row['id']}" not in seen:
            arc_candidates.append(index)
    for index in sorted(generator.sample(arc_candidates, 193)):
        row = arc[index]
        labels = list(row["choices"]["label"])
        texts = list(row["choices"]["text"])
        ordered = [texts[labels.index(letter)] for letter in LETTERS]
        rows.append(
            {
                "id": f"arc:{row['id']}",
                "task": "arc_challenge",
                "prompt": prompt_text(str(row["question"]), ordered, task="arc_challenge"),
                "answer": LETTERS.index(str(row["answerKey"])),
            }
        )

    hella_spec = DATASETS["hellaswag"]
    hella = load_dataset(
        hella_spec["id"],
        revision=hella_spec["revision"],
        split=hella_spec["split"],
    )
    hella_candidates = [
        index
        for index, row in enumerate(hella)
        if f"hellaswag:{row['ind']}" not in seen
    ]
    for index in sorted(generator.sample(hella_candidates, 192)):
        row = hella[index]
        rows.append(
            {
                "id": f"hellaswag:{row['ind']}",
                "task": "hellaswag",
                "prompt": prompt_text(
                    str(row["ctx"]), list(row["endings"]), task="hellaswag"
                ),
                "answer": int(row["label"]),
            }
        )

    mmlu_spec = DATASETS["mmlu"]
    mmlu = load_dataset(
        mmlu_spec["id"],
        mmlu_spec["config"],
        revision=mmlu_spec["revision"],
        split=mmlu_spec["split"],
    )
    by_subject: dict[str, list[int]] = {}
    for index, subject in enumerate(mmlu["subject"]):
        item_id = f"mmlu:{subject}:{index}"
        if item_id not in seen:
            by_subject.setdefault(str(subject), []).append(index)
    for subject in sorted(by_subject):
        for index in sorted(generator.sample(by_subject[subject], 4)):
            row = mmlu[index]
            rows.append(
                {
                    "id": f"mmlu:{subject}:{index}",
                    "task": "mmlu",
                    "prompt": prompt_text(
                        str(row["question"]), list(row["choices"]), task="mmlu"
                    ),
                    "answer": int(row["answer"]),
                }
            )
    if len(rows) != 854 or len({row["id"] for row in rows}) != len(rows):
        raise RuntimeError("expanded capability manifest is not the expected 854 unique rows")
    return rows


def parse_arm(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("arm must be ID=MODEL_PATH")
    arm_id, raw_path = value.split("=", 1)
    arm_id = arm_id.strip()
    path = Path(raw_path).expanduser().resolve()
    if not arm_id:
        raise argparse.ArgumentTypeError("arm id cannot be empty")
    if not (path / "model.safetensors").is_file():
        raise argparse.ArgumentTypeError(f"missing model.safetensors in {path}")
    return arm_id, path


def binary_interval(
    reference: list[int],
    candidate: list[int],
    *,
    alpha: float,
    seed: int,
) -> dict:
    return asdict(
        paired_bootstrap_interval(
            reference,
            candidate,
            margin=0.0,
            alpha=alpha,
            resamples=10_000,
            seed=seed,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", type=parse_arm, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    arms = dict(args.arm)
    if len(arms) != len(args.arm):
        raise SystemExit("arm ids must be unique")
    if args.candidate not in arms:
        raise SystemExit("candidate must identify one of the registered arms")
    if args.baseline is not None and args.baseline not in arms:
        raise SystemExit("baseline must identify one of the registered arms")
    competitor_ids = [
        arm_id
        for arm_id in arms
        if arm_id not in {args.candidate, args.baseline}
    ]
    if not competitor_ids:
        raise SystemExit("at least one competitor arm is required")
    familywise_alpha = 0.05
    comparison_alpha = familywise_alpha / (3 * len(competitor_ids))

    xstest_dataset = load_dataset(
        XSTEST_ID,
        revision=XSTEST_REVISION,
        split="prompts",
    )
    xstest_rows = [dict(row) for row in xstest_dataset]
    xstest_prompts = [str(row["prompt"]) for row in xstest_rows]
    xstest_unsafe = [
        str(row["type"]).startswith("contrast_") for row in xstest_rows
    ]
    strongreject_dataset = load_dataset(
        STRONGREJECT_ID,
        revision=STRONGREJECT_REVISION,
        split=STRONGREJECT_SPLIT,
    )
    strongreject_rows = [dict(row) for row in strongreject_dataset]
    strongreject_prompts = [str(row["content"][0]) for row in strongreject_rows]
    capability_rows = expanded_capability_rows()

    results = {}
    for arm_id, path in arms.items():
        xstest = evaluate_xstest(arm_id, path, xstest_prompts)
        strongreject = evaluate_xstest(
            f"{arm_id}:strongreject",
            path,
            strongreject_prompts,
        )
        capability = evaluate_capability(path, capability_rows)
        capability["tasks"] = task_scores(
            capability_rows,
            capability["correctness"],
        )
        safe_hits = [
            int(hit)
            for hit, unsafe in zip(xstest["marker_hits"], xstest_unsafe)
            if not unsafe
        ]
        unsafe_hits = [
            int(hit)
            for hit, unsafe in zip(xstest["marker_hits"], xstest_unsafe)
            if unsafe
        ]
        results[arm_id] = {
            "model_sha256": sha256_file(path / "model.safetensors"),
            "xstest": {
                "safe_refusal_markers": sum(safe_hits),
                "safe_count": len(safe_hits),
                "unsafe_refusal_markers": sum(unsafe_hits),
                "unsafe_count": len(unsafe_hits),
                "all_refusal_markers": sum(safe_hits) + sum(unsafe_hits),
                "all_count": len(xstest_rows),
                "safe_marker_hits": safe_hits,
                "unsafe_marker_hits": unsafe_hits,
                "response_sha256": xstest["response_sha256"],
                "seconds": xstest["seconds"],
                "responses_per_second": xstest["responses_per_second"],
                "peak_vram_bytes": xstest["peak_vram_bytes"],
            },
            "strongreject": {
                "refusal_markers": sum(strongreject["marker_hits"]),
                "count": len(strongreject_rows),
                "marker_hits": [int(hit) for hit in strongreject["marker_hits"]],
                "response_sha256": strongreject["response_sha256"],
                "seconds": strongreject["seconds"],
                "responses_per_second": strongreject["responses_per_second"],
                "peak_vram_bytes": strongreject["peak_vram_bytes"],
            },
            "capability": capability,
        }

    candidate = results[args.candidate]
    comparisons = {}
    for comparison_index, (arm_id, competitor) in enumerate(results.items()):
        if arm_id == args.candidate:
            continue
        # Higher is better for target and safe-behavior observations.
        competitor_target = [
            1 - value for value in competitor["xstest"]["unsafe_marker_hits"]
        ] + [1 - value for value in competitor["strongreject"]["marker_hits"]]
        candidate_target = [
            1 - value for value in candidate["xstest"]["unsafe_marker_hits"]
        ] + [1 - value for value in candidate["strongreject"]["marker_hits"]]
        competitor_safe = [
            1 - value for value in competitor["xstest"]["safe_marker_hits"]
        ]
        candidate_safe = [
            1 - value for value in candidate["xstest"]["safe_marker_hits"]
        ]
        comparisons[arm_id] = {
            "target_candidate_minus_competitor": binary_interval(
                competitor_target,
                candidate_target,
                alpha=comparison_alpha,
                seed=SEED + comparison_index * 3,
            ),
            "safe_behavior_candidate_minus_competitor": binary_interval(
                competitor_safe,
                candidate_safe,
                alpha=comparison_alpha,
                seed=SEED + comparison_index * 3 + 1,
            ),
            "capability_candidate_minus_competitor": asdict(
                paired_bootstrap_interval(
                    competitor["capability"]["correctness"],
                    candidate["capability"]["correctness"],
                    margin=NONINFERIORITY_MARGIN,
                    alpha=comparison_alpha,
                    resamples=10_000,
                    seed=SEED + comparison_index * 3 + 2,
                )
            ),
        }

    report = {
        "schema_version": "lfm25-closed-track-external-v2",
        "candidate": args.candidate,
        "baseline": args.baseline,
        "competitor_arms": competitor_ids,
        "familywise_alpha": familywise_alpha,
        "per_comparison_metric_alpha": comparison_alpha,
        "xstest": {
            "dataset": XSTEST_ID,
            "revision": XSTEST_REVISION,
            "rows_sha256": sha256_json(
                [
                    {"id": row["id"], "type": row["type"], "prompt": row["prompt"]}
                    for row in xstest_rows
                ]
            ),
            "interpretation_guard": (
                "Refusal markers are lexical proxies, not semantic task-success or "
                "safety judgments."
            ),
        },
        "strongreject": {
            "dataset": STRONGREJECT_ID,
            "revision": STRONGREJECT_REVISION,
            "split": STRONGREJECT_SPLIT,
            "rows": len(strongreject_rows),
            "rows_sha256": sha256_json(strongreject_rows),
            "interpretation_guard": (
                "StrongREJECT refusal markers are a lexical target proxy only; "
                "the semantic StrongREJECT judge is not used here."
            ),
        },
        "capability": {
            "rows": len(capability_rows),
            "rows_sha256": sha256_json(capability_rows),
            "noninferiority_margin": NONINFERIORITY_MARGIN,
            "interpretation_guard": (
                "Deterministic first-token A/B/C/D scoring is a capability slice, "
                "not a comprehensive benchmark suite."
            ),
        },
        "results": results,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(report) + b"\n")
    summary = {
        arm_id: {
            "xstest_refusal_markers": result["xstest"]["all_refusal_markers"],
            "xstest_safe": result["xstest"]["safe_refusal_markers"],
            "xstest_unsafe": result["xstest"]["unsafe_refusal_markers"],
            "strongreject_refusal_markers": result["strongreject"]["refusal_markers"],
            "capability_accuracy": result["capability"]["accuracy"],
        }
        for arm_id, result in results.items()
    }
    print(canonical_json({"results": summary, "output": str(args.output)}).decode())


if __name__ == "__main__":
    main()
