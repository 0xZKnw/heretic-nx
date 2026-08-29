#!/usr/bin/env python3
"""Build one bounded-memory direct-Q8 Ling-3.0-tiny candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from gguf import GGMLQuantizationType, GGUFReader

from heretic_nx.edits import (
    GGUFQ8AblationPlan,
    GGUFQ8TensorEdit,
    apply_q8_gguf_ablation,
)
from heretic_nx.hashing import canonical_json, sha256_file


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "checkpoints" / "ling3-tiny-q8" / "Ling-3.0-tiny-Q8_0.gguf"
RUN_DIR = ROOT / "runs" / "ling3-tiny-q8-direct"
AXES = RUN_DIR / "target-native-axes.safetensors"
TARGET_PATTERN = re.compile(
    r"^blk\.(?P<layer>\d+)\."
    r"(?P<kind>attn_output|ffn_down|ffn_down_exps|ffn_down_shexp)\.weight$"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--axis",
        choices=("axis.raw", "axis.uncentered", "axis.r1", "axis.r2", "axis.r4", "axis.r8"),
        default="axis.r8",
    )
    parser.add_argument("--axis-artifact", type=Path, default=AXES)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--stop-layer", type=int, default=24)
    parser.add_argument(
        "--layers",
        help="comma-separated explicit layer indices; overrides start/stop",
    )
    parser.add_argument("--include-routed-ffn", action="store_true")
    parser.add_argument("--include-shared-ffn", action="store_true")
    parser.add_argument("--include-dense-ffn", action="store_true")
    parser.add_argument("--ffn-multiplier", type=float, default=1.0)
    parser.add_argument("--no-row-norms", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.start_layer < args.stop_layer <= 24:
        raise ValueError("layers must satisfy 0 <= start < stop <= 24")
    selected_layers = None
    if args.layers is not None:
        try:
            selected_layers = {int(value) for value in args.layers.split(",")}
        except ValueError as error:
            raise ValueError("--layers must be comma-separated integers") from error
        if not selected_layers or min(selected_layers) < 0 or max(selected_layers) >= 24:
            raise ValueError("--layers values must be between 0 and 23")
    if args.beta < 0 or args.ffn_multiplier < 0:
        raise ValueError("beta and FFN multiplier must be non-negative")
    if not SOURCE.is_file() or not args.axis_artifact.is_file():
        raise RuntimeError("source Q8 GGUF or target-native axes are missing")

    reader = GGUFReader(SOURCE)
    targets = []
    skipped = []
    for tensor in reader.tensors:
        match = TARGET_PATTERN.match(tensor.name)
        if match is None:
            continue
        layer = int(match.group("layer"))
        if selected_layers is not None:
            if layer not in selected_layers:
                continue
        elif not args.start_layer <= layer < args.stop_layer:
            continue
        kind = match.group("kind")
        enabled = (
            kind == "attn_output"
            or (kind == "ffn_down_exps" and args.include_routed_ffn)
            or (kind == "ffn_down_shexp" and args.include_shared_ffn)
            or (kind == "ffn_down" and args.include_dense_ffn)
        )
        if not enabled:
            continue
        if tensor.tensor_type != GGMLQuantizationType.Q8_0:
            skipped.append({"tensor": tensor.name, "type": tensor.tensor_type.name})
            continue
        targets.append(
            GGUFQ8TensorEdit(
                tensor_name=tensor.name,
                a_key=args.axis,
                strength=args.beta * (args.ffn_multiplier if kind.startswith("ffn") else 1.0),
                preserve_row_norms=not args.no_row_norms,
            )
        )
    if not targets:
        raise RuntimeError("candidate profile selected no editable tensors")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    plan_path = RUN_DIR / f"{args.label}.plan.json"
    report_path = RUN_DIR / f"{args.label}.merge.json"
    plan = GGUFQ8AblationPlan(
        source_sha256=sha256_file(SOURCE),
        tensor_artifact_sha256=sha256_file(args.axis_artifact),
        edits=tuple(targets),
    )
    plan.write(plan_path)
    report = apply_q8_gguf_ablation(
        SOURCE,
        args.output,
        plan_path,
        args.axis_artifact,
        force=args.force,
    )
    report["candidate"] = {
        "label": args.label,
        "axis": args.axis,
        "beta": args.beta,
        "start_layer": args.start_layer,
        "stop_layer": args.stop_layer,
        "selected_layers": sorted(selected_layers) if selected_layers is not None else None,
        "include_routed_ffn": args.include_routed_ffn,
        "include_shared_ffn": args.include_shared_ffn,
        "include_dense_ffn": args.include_dense_ffn,
        "ffn_multiplier": args.ffn_multiplier,
        "preserve_row_norms": not args.no_row_norms,
        "active_tensors": len(targets),
        "skipped_non_q8": skipped,
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
