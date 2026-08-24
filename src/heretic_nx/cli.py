"""Minimal CLI for the reproducibility-first milestone."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import torch

from . import __version__
from .runtime.token_budget import OperationBudgetRegistry, cuda_memory_caps


def doctor(backend: str) -> dict[str, object]:
    report: dict[str, object] = {
        "heretic_nx": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "requested_backend": backend,
        "cuda_available": torch.cuda.is_available(),
        "libraries": {
            name: importlib.metadata.version(name)
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
    args = parser.parse_args()

    if args.command == "doctor":
        report = doctor(args.backend)
        rendered = json.dumps(report, indent=2, sort_keys=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)


if __name__ == "__main__":
    main()
