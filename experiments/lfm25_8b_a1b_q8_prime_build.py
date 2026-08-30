#!/usr/bin/env python3
"""Build one site-ranked PRIME candidate directly in the target Q8 GGUF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from gguf import GGMLQuantizationType, GGUFReader
from safetensors import safe_open

from experiments.lfm25_8b_a1b_prime_sites import OUTPUT as OPERATORS
from experiments.lfm25_8b_a1b_prime_sites import REPORT as RANKING_REPORT
from experiments.lfm25_8b_a1b_q8_build import ROOT, RUN_DIR, SOURCE
from heretic_nx.edits import (
    GGUFQ8AblationPlan,
    GGUFQ8TensorEdit,
    apply_q8_gguf_ablation,
)
from heretic_nx.hashing import canonical_json, sha256_file
from heretic_nx.hashing import sha256_json


def _artifact_record(
    value: object,
    *,
    expected_path: Path,
    field_name: str,
    actual_sha256: str | None = None,
) -> str:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{field_name} provenance is missing")
    path = Path(str(value.get("path", "")))
    if path.resolve() != expected_path.resolve() or not expected_path.is_file():
        raise RuntimeError(f"{field_name} path does not match the current artifact")
    actual = actual_sha256 or sha256_file(expected_path)
    if value.get("sha256") != actual:
        raise RuntimeError(f"{field_name} hash does not match the current artifact")
    return actual


def load_verified_ranking(
    *,
    source_path: Path = SOURCE,
    operators_path: Path = OPERATORS,
    ranking_path: Path = RANKING_REPORT,
    _source_sha256: str | None = None,
    _operators_sha256: str | None = None,
) -> dict[str, Any]:
    """Load only a v2 ranking fully bound to current source and factors."""

    if not source_path.is_file() or not operators_path.is_file():
        raise RuntimeError("current Q8 source or PRIME operator artifact is missing")
    current_source_sha256 = _source_sha256 or sha256_file(source_path)
    current_operators_sha256 = _operators_sha256 or sha256_file(operators_path)
    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    if ranking.get("schema_version") != "lfm25-8b-a1b-prime-sites-v2":
        raise RuntimeError("PRIME ranking predates leak-safe geometry provenance")
    operator_sha256 = _artifact_record(
        ranking.get("operator_artifact"),
        expected_path=operators_path,
        field_name="operator artifact",
        actual_sha256=current_operators_sha256,
    )
    provenance = ranking.get("geometry_provenance")
    activations = ranking.get("activations")
    if not isinstance(provenance, Mapping) or not isinstance(activations, Mapping):
        raise RuntimeError("PRIME ranking geometry provenance is missing")
    if (
        provenance.get("source_sha256") != current_source_sha256
        or Path(str(provenance.get("source", ""))).resolve()
        != source_path.resolve()
        or activations.get("manifest_sha256")
        != provenance.get("activation_manifest_sha256")
    ):
        raise RuntimeError("PRIME ranking belongs to another Q8 source or geometry")
    with safe_open(operators_path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        factor_keys = set(handle.keys())
        accepted = ranking.get("accepted")
        if not isinstance(accepted, list) or not accepted:
            raise RuntimeError("PRIME ranking contains no accepted sites")
        site_ids: set[str] = set()
        bound_keys: set[str] = set()
        for row in accepted:
            if not isinstance(row, Mapping):
                raise RuntimeError("PRIME ranking site is not an object")
            site_id = str(row.get("site_id", ""))
            a_key = str(row.get("factor_a_key", ""))
            b_key = str(row.get("factor_b_key", ""))
            if (
                not site_id
                or site_id in site_ids
                or not a_key
                or not b_key
                or a_key in bound_keys
                or b_key in bound_keys
                or a_key not in factor_keys
                or b_key not in factor_keys
            ):
                raise RuntimeError("PRIME ranking has duplicate or missing factor bindings")
            a = handle.get_tensor(a_key)
            b = handle.get_tensor(b_key)
            if (
                a.ndim != 2
                or b.ndim != 2
                or a.shape != b.shape
                or int(row.get("rank", -1)) != a.shape[1]
            ):
                raise RuntimeError(f"PRIME factor shape mismatch for {site_id}")
            site_ids.add(site_id)
            bound_keys.update((a_key, b_key))
    expected_metadata = {
        "activation_manifest_sha256": provenance.get(
            "activation_manifest_sha256"
        ),
        "safe_split_manifest_sha256": provenance.get(
            "safe_split_manifest_sha256"
        ),
        "target_split_manifest_sha256": provenance.get(
            "target_split_manifest_sha256"
        ),
        "source_sha256": current_source_sha256,
        "sites_sha256": provenance.get("sites_sha256"),
        "activations_sha256": activations.get("sha256"),
    }
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise RuntimeError("PRIME factors and ranking provenance disagree")
    if operator_sha256 != current_operators_sha256:
        raise RuntimeError("PRIME operator artifact changed during validation")
    return ranking


def load_verified_prime_merge(
    merge_path: Path,
    *,
    source_path: Path = SOURCE,
    operators_path: Path = OPERATORS,
    ranking_path: Path = RANKING_REPORT,
    verify_output: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify a built PRIME teacher and every upstream artifact binding."""

    source_sha256 = sha256_file(source_path)
    operators_sha256 = sha256_file(operators_path)
    ranking = load_verified_ranking(
        source_path=source_path,
        operators_path=operators_path,
        ranking_path=ranking_path,
        _source_sha256=source_sha256,
        _operators_sha256=operators_sha256,
    )
    merge = json.loads(merge_path.read_text(encoding="utf-8"))
    if merge.get("schema_version") != "gguf-quantized-static-merge-report-v3":
        raise RuntimeError("unsupported PRIME teacher merge report")
    candidate = merge.get("candidate")
    if not isinstance(candidate, Mapping):
        raise RuntimeError("PRIME teacher candidate provenance is missing")
    ranking_sha256 = sha256_file(ranking_path)
    geometry_sha256 = sha256_json(ranking["geometry_provenance"])
    if (
        candidate.get("ranking_report_sha256") != ranking_sha256
        or Path(str(candidate.get("ranking_report", ""))).resolve()
        != ranking_path.resolve()
        or candidate.get("operator_artifact_sha256") != operators_sha256
        or candidate.get("geometry_provenance_sha256") != geometry_sha256
    ):
        raise RuntimeError("PRIME teacher is not bound to the current ranking")
    _artifact_record(
        merge.get("source"),
        expected_path=source_path,
        field_name="teacher source",
        actual_sha256=source_sha256,
    )
    _artifact_record(
        merge.get("tensor_artifact"),
        expected_path=operators_path,
        field_name="teacher operator artifact",
        actual_sha256=operators_sha256,
    )
    selected = candidate.get("selected")
    if (
        not isinstance(selected, list)
        or not selected
        or candidate.get("selected_sites_sha256") != sha256_json(selected)
    ):
        raise RuntimeError("PRIME teacher selected-site binding is invalid")
    accepted_by_site = {
        str(row["site_id"]): row for row in ranking["accepted"]
    }
    for row in selected:
        if (
            not isinstance(row, Mapping)
            or accepted_by_site.get(str(row.get("site_id", "")))
            != {key: value for key, value in row.items() if key != "tensor_name"}
        ):
            raise RuntimeError("PRIME teacher site no longer matches its ranking")
    plan_record = merge.get("plan")
    if not isinstance(plan_record, Mapping):
        raise RuntimeError("PRIME teacher plan provenance is missing")
    plan_path = Path(str(plan_record.get("path", "")))
    if (
        not plan_path.is_file()
        or plan_record.get("sha256") != sha256_file(plan_path)
    ):
        raise RuntimeError("PRIME teacher plan hash mismatch")
    plan = GGUFQ8AblationPlan.read(plan_path)
    if (
        plan.source_sha256 != source_sha256
        or plan.tensor_artifact_sha256 != operators_sha256
        or len(plan.edits) != len(selected)
    ):
        raise RuntimeError("PRIME teacher plan belongs to another artifact")
    beta = float(candidate.get("beta", -1.0))
    for edit, row in zip(plan.edits, selected, strict=True):
        if (
            edit.tensor_name != row.get("tensor_name")
            or edit.a_key != row.get("factor_a_key")
            or edit.b_key != row.get("factor_b_key")
            or edit.strength != beta
            or edit.preserve_row_norms
        ):
            raise RuntimeError("PRIME teacher plan and selected sites disagree")
    if verify_output:
        output = merge.get("output")
        if not isinstance(output, Mapping):
            raise RuntimeError("PRIME teacher output provenance is missing")
        output_path = Path(str(output.get("path", "")))
        _artifact_record(
            output,
            expected_path=output_path,
            field_name="teacher output",
        )
    return merge, ranking


def tensor_name(row: dict[str, object]) -> str:
    layer = int(row["layer"])
    family = str(row["family"])
    if family == "gqa":
        return f"blk.{layer}.attn_output.weight"
    if family == "liv":
        return f"blk.{layer}.shortconv.out_proj.weight"
    if family == "ffn":
        suffix = "ffn_down.weight" if layer < 2 else "ffn_down_exps.weight"
        return f"blk.{layer}.{suffix}"
    raise ValueError(f"unsupported PRIME family: {family}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument(
        "--max-safe-drift",
        type=float,
        help="discard PRIME sites whose calibration safe-drift exceeds this value",
    )
    parser.add_argument(
        "--families",
        help="comma-separated PRIME families to retain (ffn,gqa,liv)",
    )
    parser.add_argument(
        "--exclude-sites",
        help="comma-separated semantic site ids to discard",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.k <= 0 or args.beta < 0:
        raise ValueError("k must be positive and beta must be non-negative")
    if args.max_safe_drift is not None and args.max_safe_drift <= 0:
        raise ValueError("max-safe-drift must be positive")
    families = (
        {value.strip() for value in args.families.split(",") if value.strip()}
        if args.families
        else None
    )
    if families is not None and not families <= {"ffn", "gqa", "liv"}:
        raise ValueError("families must contain only ffn, gqa, or liv")
    excluded = (
        {value.strip() for value in args.exclude_sites.split(",") if value.strip()}
        if args.exclude_sites
        else set()
    )

    ranking_report = load_verified_ranking()
    ranking = ranking_report["accepted"]
    reader = GGUFReader(SOURCE)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    editable = []
    filtered = []
    skipped = []
    for row in ranking:
        if (
            args.max_safe_drift is not None
            and float(row["safe_proxy_drift"]) > args.max_safe_drift
        ):
            filtered.append({"site_id": row["site_id"], "reason": "safe-drift"})
            continue
        if families is not None and str(row["family"]) not in families:
            filtered.append({"site_id": row["site_id"], "reason": "family"})
            continue
        if str(row["site_id"]) in excluded:
            filtered.append({"site_id": row["site_id"], "reason": "explicit-exclusion"})
            continue
        name = tensor_name(row)
        tensor = tensors.get(name)
        if tensor is None or tensor.tensor_type != GGMLQuantizationType.Q8_0:
            skipped.append(
                {
                    "site_id": row["site_id"],
                    "tensor_name": name,
                    "reason": "missing" if tensor is None else tensor.tensor_type.name,
                }
            )
            continue
        editable.append({**row, "tensor_name": name})
    selected = editable[: min(args.k, len(editable))]
    if not selected:
        raise RuntimeError("the PRIME ranking exposes no editable Q8 sites")
    edits = tuple(
        GGUFQ8TensorEdit(
            tensor_name=str(row["tensor_name"]),
            a_key=str(row["factor_a_key"]),
            b_key=str(row["factor_b_key"]),
            strength=args.beta,
            preserve_row_norms=False,
        )
        for row in selected
    )
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    plan_path = RUN_DIR / f"{args.label}.plan.json"
    report_path = RUN_DIR / f"{args.label}.merge.json"
    plan = GGUFQ8AblationPlan(
        source_sha256=sha256_file(SOURCE),
        tensor_artifact_sha256=sha256_file(OPERATORS),
        edits=edits,
    )
    plan.write(plan_path)
    report = apply_q8_gguf_ablation(
        SOURCE,
        args.output,
        plan_path,
        OPERATORS,
        force=args.force,
    )
    report["candidate"] = {
        "label": args.label,
        "k": args.k,
        "beta": args.beta,
        "active_sites": len(selected),
        "preserve_row_norms": False,
        "max_safe_drift": args.max_safe_drift,
        "families": sorted(families) if families is not None else None,
        "excluded_sites": sorted(excluded),
        "selected": selected,
        "filtered": filtered,
        "skipped_non_q8": skipped,
        "ranking_report": str(RANKING_REPORT),
        "ranking_report_sha256": sha256_file(RANKING_REPORT),
        "operator_artifact_sha256": sha256_file(OPERATORS),
        "geometry_provenance_sha256": sha256_json(
            ranking_report["geometry_provenance"]
        ),
        "selected_sites_sha256": sha256_json(selected),
    }
    report_path.write_bytes(canonical_json(report) + b"\n")
    print(
        json.dumps(
            {
                "candidate": report["candidate"],
                "output": report["output"],
                "report": str(report_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
