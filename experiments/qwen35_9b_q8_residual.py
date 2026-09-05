#!/usr/bin/env python3
"""Shared residual-stream direction diagnostic using exact native Q8 states."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from gguf import GGUFReader
from safetensors.torch import load_file, save_file

from heretic_nx.edits import GGUFQuantizedAblationPlan, GGUFQuantizedTensorEdit, apply_quantized_gguf_ablation
from heretic_nx.edits.gguf_codecs import GGUFQuantizationCodecRegistry
from heretic_nx.geometry.contrastive import fit_contrastive_axis
from heretic_nx.hashing import sha256_file, sha256_json
from qwen35_9b_q8_factors import fit_detector
from qwen35_9b_q8_geometry import ROOT, RUN, BASE, BASE_SHA, write

EXE = ROOT / "build/llama.cpp-native/bin/llama_capture_named"
OUT = RUN / "residual64.capture"


def capture():
    if sha256_file(BASE) != BASE_SHA:
        raise RuntimeError("baseline changed")
    OUT.mkdir(exist_ok=False)
    tokens = RUN / "geometry64.capture/tokens.txt"
    started = time.monotonic()
    with (OUT / "collector.log").open("x") as log:
        subprocess.run([str(EXE), str(BASE), str(tokens), str(OUT),
                        ",".join(f"@l_out-{i}" for i in range(32))],
                       check=True, stdout=log, stderr=subprocess.STDOUT)
    payload, diagnostics = {}, []
    for i in range(32):
        values = np.stack([np.fromfile(OUT / f"row{r}.site{i}.f32", dtype="<f4") for r in range(128)])
        if values.shape != (128, 4096) or not np.isfinite(values).all():
            raise RuntimeError("invalid residual capture")
        safe, target = torch.from_numpy(values[:64]), torch.from_numpy(values[64:])
        axis = fit_contrastive_axis(safe, target, remove_safe_mean=True)
        payload[f"layer{i}"] = axis.axis[:, None].contiguous()
        diagnostics.append({"layer": i, "fold_cosine_minimum": axis.fold_cosine_minimum,
            "safe_projection_rms": float((safe @ axis.axis).square().mean().sqrt()),
            "target_projection_rms": float((target @ axis.axis).square().mean().sqrt())})
    for i in range(128):
        a = np.fromfile(OUT / f"row{i}.logits.f32", dtype="<f4")
        b = np.fromfile(RUN / "geometry64.capture" / f"row{i}.logits.f32", dtype="<f4")
        if not np.array_equal(a, b):
            raise RuntimeError(f"named capture changed logits on row {i}")
    axes = RUN / "residual64.axes.safetensors"
    save_file(payload, axes, metadata={"source_sha256": BASE_SHA})
    write(RUN / "residual64.capture.json", {"source_sha256": BASE_SHA, "rows": 128,
          "capture_executable_sha256": sha256_file(EXE), "tokens_sha256": sha256_file(tokens),
          "axes_sha256": sha256_file(axes), "all_logits_bit_identical": True,
          "seconds": time.monotonic() - started, "diagnostics": diagnostics})
    print(json.dumps({"captured": 128, "all_logits_bit_identical": True}), flush=True)


def build(args):
    torch.set_num_threads(4)
    cap = json.loads((RUN / "residual64.capture.json").read_text())
    axis_path = RUN / "residual64.axes.safetensors"
    if sha256_file(axis_path) != cap["axes_sha256"] or not cap["all_logits_bit_identical"]:
        raise RuntimeError("unverified residual axes")
    axis = load_file(axis_path)[f"layer{args.layer}"]
    prep = json.loads((RUN / "geometry64.preparation.json").read_text())
    input_path = RUN / "geometry64.inputs.safetensors"
    input_report = json.loads((RUN / "geometry64.capture.json").read_text())
    if sha256_file(input_path) != input_report["inputs_sha256"]:
        raise RuntimeError("input capture changed")
    inputs = load_file(input_path)
    tensors = {t.name: t for t in GGUFReader(BASE).tensors}
    payload = {"axis": axis}
    edits, diagnostics = [], []
    with GGUFQuantizationCodecRegistry() as codec:
        for i, row in enumerate(prep["sites"]):
            if not args.first <= row["layer"] <= args.last:
                continue
            if args.family == "mixer" and row["family"] == "ffn_down":
                continue
            if args.family == "ffn" and row["family"] != "ffn_down":
                continue
            options = {}
            if args.protection is not None:
                tensor = tensors[row["tensor_name"]]
                w = torch.from_numpy(codec.dequantize_rows(tensor.data, tensor.tensor_type, row["input_dim"]))
                teacher = (w.T @ axis).flatten()
                right, evidence = fit_detector(inputs[f"site{i}.target"], inputs[f"site{i}.safe"], teacher, args.protection)
                payload[f"site{i}.right"] = right[:, None].contiguous()
                options["right_key"] = f"site{i}.right"
                diagnostics.append({"site": i, **evidence})
                del w
            edits.append(GGUFQuantizedTensorEdit(tensor_name=row["tensor_name"], expected_quantization="Q8_0",
                a_key="axis", strength=args.beta, preserve_row_norms=False, **options))
    factors = RUN / f"{args.label}.safetensors"
    plan_path = RUN / f"{args.label}.plan.json"
    if factors.exists() or plan_path.exists():
        raise FileExistsError("candidate label already exists")
    save_file(payload, factors, metadata={"source_sha256": BASE_SHA, "residual_capture_sha256": sha256_json(cap)})
    plan = GGUFQuantizedAblationPlan(source_sha256=BASE_SHA, tensor_artifact_sha256=sha256_file(factors), edits=tuple(edits))
    plan.write(plan_path)
    artifact = ROOT / "outputs" / f"qwen35-9b-q8-{args.label}.gguf"
    report = apply_quantized_gguf_ablation(BASE, artifact, plan_path, factors)
    write(RUN / f"{args.label}.build.json", {**report, "research": {"axis_layer": args.layer,
          "first_layer": args.first, "last_layer": args.last, "family": args.family,
          "protection": args.protection, "solver_diagnostics": diagnostics}})
    print(json.dumps({"artifact": str(artifact), "sites": len(edits), "sha256": sha256_file(artifact)}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("capture", "build"))
    p.add_argument("--label")
    p.add_argument("--layer", type=int, choices=range(32), default=16)
    p.add_argument("--first", type=int, choices=range(32), default=0)
    p.add_argument("--last", type=int, choices=range(32), default=31)
    p.add_argument("--family", choices=("all", "mixer", "ffn"), default="all")
    p.add_argument("--protection", type=float)
    p.add_argument("--beta", type=float, default=1.0)
    args = p.parse_args()
    if args.first > args.last:
        p.error("invalid layer interval")
    if args.command == "capture":
        capture()
    else:
        if not args.label or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in args.label):
            p.error("invalid label")
        build(args)
