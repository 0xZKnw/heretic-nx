"""Minimal CLI for the reproducibility-first milestone."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import torch

from . import __version__
from .runtime.token_budget import OperationBudgetRegistry, cuda_memory_caps
from .benchmark.closed_track import (
    ArmObservations,
    ClosedTrackRegistration,
    evaluate_closed_track,
)
from .hashing import canonical_json


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def doctor(backend: str) -> dict[str, object]:
    report: dict[str, object] = {
        "heretic_nx": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "requested_backend": backend,
        "cuda_available": torch.cuda.is_available(),
        "libraries": {
            name: _package_version(name)
            for name in ("transformers", "bitsandbytes", "safetensors", "scipy")
        },
        "torch_cuda_runtime": torch.version.cuda,
        "operation_budgets": {
            operation: {
                "batch_size": budget.batch_controller.batch_size,
                "token_budget": budget.token_budget,
            }
            for operation, budget in OperationBudgetRegistry.defaults().budgets.items()
        },
    }
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        report["cuda"] = {
            "device": torch.cuda.get_device_name(0),
            "free_bytes": free,
            "total_bytes": total,
            "capability": list(torch.cuda.get_device_capability(0)),
            "memory_caps_bytes": dict(zip(("soft", "hard"), cuda_memory_caps(free))),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(prog="hnx")
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor_parser = subparsers.add_parser("doctor")
    doctor_parser.add_argument("--backend", default="auto")
    doctor_parser.add_argument("--output", type=Path)
    benchmark_parser = subparsers.add_parser("benchmark-closed")
    benchmark_parser.add_argument("--registration", type=Path, required=True)
    benchmark_parser.add_argument("--observations", type=Path, required=True)
    benchmark_parser.add_argument("--output", type=Path, required=True)
    benchmark_parser.add_argument("--seed", type=int, default=0)
    inspect_q8_parser = subparsers.add_parser(
        "inspect-q8",
        help="list directly editable Q8_0 tensors in a GGUF",
    )
    inspect_q8_parser.add_argument("--input", type=Path, required=True)
    inspect_q8_parser.add_argument("--output", type=Path)
    ablate_q8_parser = subparsers.add_parser(
        "abliterate-q8",
        help="apply a low-rank ablation plan directly to a Q8_0 GGUF",
    )
    ablate_q8_parser.add_argument("--input", type=Path, required=True)
    ablate_q8_parser.add_argument("--output", type=Path)
    ablate_q8_parser.add_argument("--plan", type=Path, required=True)
    ablate_q8_parser.add_argument("--tensors", type=Path, required=True)
    ablate_q8_parser.add_argument("--report", type=Path)
    ablate_q8_parser.add_argument("--dry-run", action="store_true")
    ablate_q8_parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.command == "doctor":
        report = doctor(args.backend)
        rendered = json.dumps(report, indent=2, sort_keys=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    elif args.command == "benchmark-closed":
        registration = ClosedTrackRegistration.model_validate_json(
            args.registration.read_bytes()
        )
        payload = json.loads(args.observations.read_text(encoding="utf-8"))
        observations = {
            arm_id: ArmObservations.model_validate(value)
            for arm_id, value in payload.items()
        }
        result = evaluate_closed_track(registration, observations, seed=args.seed)
        report = {
            "schema_version": "closed-track-report-v1",
            **asdict(result),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical_json(report) + b"\n")
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "inspect-q8":
        from .edits.gguf_q8 import inspect_q8_gguf

        report = inspect_q8_gguf(args.input)
        rendered = json.dumps(report, indent=2, sort_keys=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    elif args.command == "abliterate-q8":
        from .edits.gguf_q8 import apply_q8_gguf_ablation

        if not args.dry_run and args.output is None:
            parser.error("abliterate-q8 requires --output unless --dry-run is used")
        report = apply_q8_gguf_ablation(
            args.input,
            args.output,
            args.plan,
            args.tensors,
            dry_run=args.dry_run,
            force=args.force,
        )
        rendered = json.dumps(report, indent=2, sort_keys=True)
        report_path = args.report
        if report_path is None and args.output is not None and not args.dry_run:
            report_path = args.output.with_suffix(args.output.suffix + ".hnx.json")
        if report_path is not None:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)


if __name__ == "__main__":
    main()
