#!/usr/bin/env python3
"""Native Q8-only geometry collection and protected sparse edit preparation.

Training geometry is disjoint from the 104 development rows. No BF16 model is
loaded; only one projection matrix at a time is dequantized on CPU for fitting.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from datasets import load_dataset
from gguf import GGUFReader
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from heretic_nx.data.research_splits import (
    build_research_split, subset_research_split, verify_manifest_texts,
)
from heretic_nx.edits import GGUFQuantizedAblationPlan, GGUFQuantizedTensorEdit, apply_quantized_gguf_ablation
from heretic_nx.edits.gguf_codecs import GGUFQuantizationCodecRegistry
from heretic_nx.eval.native_logits import attest_tokenizer_assets, collect_native_raw_logits
from heretic_nx.geometry.contrastive import fit_contrastive_axis
from heretic_nx.hashing import canonical_json, sha256_file, sha256_json
from qwen35_9b_q8_factors import fit_detector
import qwen35_9b_q8_eval as refusal
import qwen35_9b_q8_kl as kl

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/qwen35-9b-q8"
BASE = ROOT / "checkpoints/qwen35-9b-gguf/Qwen3.5-9B-Q8_0.gguf"
BASE_SHA = "809626574d0cb43d4becfa56169980da2bb448f2299270f7be443cb89d0a6ae4"
EXECUTABLE = ROOT / "build/llama.cpp-native/bin/llama_capture_weight_inputs_alltokens"
FACTORS = RUN / "geometry64.factors.safetensors"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value) + b"\n")


def sites():
    tensors = {t.name: t for t in GGUFReader(BASE).tensors}
    result = []
    for layer in range(32):
        family = "attn_output" if layer % 4 == 3 else "ssm_out"
        for suffix in (family, "ffn_down"):
            name = f"blk.{layer}.{suffix}.weight"
            t = tensors[name]
            shape = tuple(reversed(t.shape.tolist()))
            expected = (4096, 12288 if suffix == "ffn_down" else 4096)
            if t.tensor_type.name != "Q8_0" or shape != expected:
                raise RuntimeError(f"unexpected projection: {name}, {shape}")
            result.append({"tensor_name": name, "layer": layer, "family": suffix,
                           "input_dim": shape[1], "output_dim": shape[0]})
    return result


def prepare():
    if sha256_file(BASE) != BASE_SHA:
        raise RuntimeError("baseline Q8 hash mismatch")
    tokenizer = AutoTokenizer.from_pretrained(refusal.engine.TOKENIZER_PATH, local_files_only=True)
    groups = {}
    token_rows = []
    for label, dataset, revision in (
        ("safe", kl.engine.GOOD_DATASET, kl.engine.GOOD_REVISION),
        ("target", refusal.engine.BAD_DATASET, refusal.engine.BAD_REVISION),
    ):
        pool = [str(row["text"]) for row in load_dataset(dataset, revision=revision, split="train[:400]")]
        manifest = build_research_split(pool, purpose="geometry", dataset_id=dataset,
                                       revision=revision, source_split="train", seed=20260905)
        manifest = subset_research_split(manifest, tuple(range(64)))
        texts = verify_manifest_texts(manifest, pool)
        rendered = refusal.engine.render(tokenizer, texts)
        tokens = [tokenizer.encode(s, add_special_tokens=False) for s in rendered]
        groups[label] = {"split_manifest": manifest.to_dict(), "split_sha256": manifest.sha256,
                         "token_sha256": sha256_json(tokens), "count": len(tokens)}
        token_rows.extend(tokens)
    value = {"schema_version": "qwen35-9b-q8-geometry-v1", "source_sha256": BASE_SHA,
             "sites": sites(), "groups": groups, "tokens": token_rows,
             "capture_executable_sha256": sha256_file(EXECUTABLE),
             "position": "last_prompt_token", "memory_reset": "llama_memory_clear(true) per row"}
    path = RUN / "geometry64.preparation.json"
    if path.exists() and json.loads(path.read_text()) != value:
        raise RuntimeError("refusing to replace different geometry preparation")
    write(path, value)
    print(json.dumps({"prepared_rows": len(token_rows), "sites": len(value["sites"])}), flush=True)


def capture():
    prep = json.loads((RUN / "geometry64.preparation.json").read_text())
    if sha256_file(BASE) != prep["source_sha256"] or sha256_file(EXECUTABLE) != prep["capture_executable_sha256"]:
        raise RuntimeError("capture inputs changed")
    out = RUN / "geometry64.capture"
    out.mkdir(exist_ok=False)
    token_path = out / "tokens.txt"
    token_path.write_text("".join(" ".join(map(str, row)) + "\n" for row in prep["tokens"]))
    started = time.monotonic()
    with (out / "collector.log").open("x") as log:
        subprocess.run([str(EXECUTABLE), str(BASE), str(token_path), str(out),
                        ",".join(s["tensor_name"] for s in prep["sites"])],
                       check=True, stdout=log, stderr=subprocess.STDOUT)
    payload = {}
    for site, entry in enumerate(prep["sites"]):
        values = np.stack([np.fromfile(out / f"row{row}.site{site}.f32", dtype="<f4")
                           for row in range(128)])
        if values.shape != (128, entry["input_dim"]) or not np.isfinite(values).all():
            raise RuntimeError(f"invalid capture for site {site}")
        payload[f"site{site}.safe"] = torch.from_numpy(values[:64].copy())
        payload[f"site{site}.target"] = torch.from_numpy(values[64:].copy())
    target = RUN / "geometry64.inputs.safetensors"
    if target.exists():
        raise FileExistsError(target)
    save_file(payload, target, metadata={"preparation_sha256": sha256_json(prep)})
    write(RUN / "geometry64.capture.json", {
        "source_sha256": sha256_file(BASE), "preparation_sha256": sha256_json(prep),
        "inputs_sha256": sha256_file(target), "seconds": time.monotonic() - started,
        "files": {p.name: sha256_file(p) for p in sorted(out.glob("*.f32"))},
    })
    print(json.dumps({"captured_rows": 128, "seconds": time.monotonic() - started}), flush=True)


def fit():
    torch.set_num_threads(4)
    prep = json.loads((RUN / "geometry64.preparation.json").read_text())
    cap = json.loads((RUN / "geometry64.capture.json").read_text())
    control = json.loads((RUN / "geometry64.control.json").read_text())
    if control.get("passed") is not True or control.get("capture_sha256") != sha256_json(cap):
        raise RuntimeError("capture logits control missing or invalid")
    inputs = RUN / "geometry64.inputs.safetensors"
    if cap["preparation_sha256"] != sha256_json(prep) or cap["inputs_sha256"] != sha256_file(inputs):
        raise RuntimeError("capture provenance mismatch")
    if sha256_file(BASE) != BASE_SHA or cap["source_sha256"] != BASE_SHA:
        raise RuntimeError("capture source mismatch")
    values = load_file(inputs)
    tensors = {t.name: t for t in GGUFReader(BASE).tensors}
    payload, ranking = {}, []
    with GGUFQuantizationCodecRegistry() as codec:
        for i, entry in enumerate(prep["sites"]):
            tensor = tensors[entry["tensor_name"]]
            w = torch.from_numpy(codec.dequantize_rows(tensor.data, tensor.tensor_type, entry["input_dim"]))
            safe, target = values[f"site{i}.safe"], values[f"site{i}.target"]
            safe_out, target_out = safe @ w.T, target @ w.T
            evidence = fit_contrastive_axis(safe_out, target_out, folds=3, remove_safe_mean=True)
            axis = evidence.axis.contiguous()
            teacher = (w.T @ axis).contiguous()
            payload[f"site{i}.axis"] = axis[:, None].contiguous()
            payload[f"site{i}.raw"] = teacher[:, None].contiguous()
            row = {**entry, "index": i, "fold_cosine_minimum": evidence.fold_cosine_minimum,
                   "protected": {}}
            for strength in (1, 10, 100):
                right, diagnostics = fit_detector(target, safe, teacher, protected_weight=strength)
                payload[f"site{i}.l{strength}"] = right[:, None].contiguous()
                safe_rms = float((safe @ right).square().mean().sqrt())
                target_rms = float((target @ right).square().mean().sqrt())
                relative_effect = target_rms / float(target_out.square().sum(1).mean().sqrt())
                score = target_rms / max(safe_rms, 1e-8) * relative_effect * max(evidence.fold_cosine_minimum, 0)
                row["protected"][str(strength)] = {"score": score, "safe_rms": safe_rms,
                    "target_rms": target_rms, "relative_target_effect": relative_effect,
                    **diagnostics}
            ranking.append(row)
            print(json.dumps({"fitted_site": i, "name": entry["tensor_name"]}), flush=True)
            del w
    if FACTORS.exists():
        raise FileExistsError(FACTORS)
    save_file(payload, FACTORS, metadata={"source_sha256": BASE_SHA, "capture_sha256": sha256_json(cap)})
    write(RUN / "geometry64.fit.json", {"source_sha256": BASE_SHA, "factors_sha256": sha256_file(FACTORS),
          "capture_sha256": sha256_json(cap), "control_sha256": sha256_json(control), "ranking": ranking})


def control():
    prep = json.loads((RUN / "geometry64.preparation.json").read_text())
    cap = json.loads((RUN / "geometry64.capture.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(refusal.engine.TOKENIZER_PATH, local_files_only=True)
    identity = attest_tokenizer_assets(refusal.engine.TOKENIZER_PATH, vocab_size=248320,
        tokenizer_class=f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}")
    indices = [0, 64]
    context = max(32, max(map(len, prep["tokens"])))
    result = collect_native_raw_logits(token_rows=[prep["tokens"][i] for i in indices],
        tokenizer_identity=identity, model_path=BASE, output_path=RUN / "geometry64.control.raw.bin",
        schema_version="qwen35-9b-q8-capture-control-v1", label="geometry64-control", model_alias="base-q8",
        executable_path=kl.engine.NATIVE_EXECUTABLE, runtime_library_dirs=[kl.engine.NATIVE_RUNTIME_DIR],
        context_size=context, batch_size=context, ubatch_size=context)
    actual = np.fromfile(result.data_path, dtype="<f4").reshape(2, 248320)
    expected = np.stack([np.fromfile(RUN / "geometry64.capture" / f"row{i}.logits.f32", dtype="<f4") for i in indices])
    maximum = float(np.max(np.abs(actual - expected)))
    passed = bool(np.isfinite(actual).all() and np.isfinite(expected).all()
                  and np.allclose(actual, expected, rtol=1e-6, atol=1e-4))
    report = {"passed": passed, "maximum_absolute_logit_difference": maximum,
              "bit_identical": bool(np.array_equal(actual, expected)), "indices": indices,
              "capture_sha256": sha256_json(cap), "raw_progress_sha256": sha256_json(result.progress)}
    write(RUN / "geometry64.control.json", report)
    print(json.dumps(report), flush=True)
    if not passed:
        raise RuntimeError("capture instrumentation changed output logits")


def build(args):
    fit_report = json.loads((RUN / "geometry64.fit.json").read_text())
    if sha256_file(FACTORS) != fit_report["factors_sha256"]:
        raise RuntimeError("factor hash mismatch")
    rows = sorted(fit_report["ranking"], key=lambda r: r["protected"][str(args.protection)]["score"], reverse=True)
    if args.sites:
        indices = [int(s) for s in args.sites.split(",")]
        if len(set(indices)) != len(indices) or any(i < 0 or i >= 64 for i in indices):
            raise ValueError("invalid site indices")
        rows = [r for r in rows if r["index"] in indices]
    else:
        rows = rows[:args.top]
    plan = GGUFQuantizedAblationPlan(source_sha256=BASE_SHA, tensor_artifact_sha256=fit_report["factors_sha256"],
        edits=tuple(GGUFQuantizedTensorEdit(tensor_name=r["tensor_name"], expected_quantization="Q8_0",
            a_key=f"site{r['index']}.axis", right_key=f"site{r['index']}.l{args.protection}",
            strength=args.beta, preserve_row_norms=False) for r in rows))
    path = RUN / f"{args.label}.plan.json"
    if path.exists():
        raise FileExistsError(path)
    plan.write(path)
    output = ROOT / "outputs" / f"qwen35-9b-q8-{args.label}.gguf"
    report = apply_quantized_gguf_ablation(BASE, output, path, FACTORS)
    write(RUN / f"{args.label}.build.json", report)
    print(json.dumps({"output": str(output), "sites": [r["index"] for r in rows],
                      "sha256": sha256_file(output)}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("prepare", "capture", "control", "fit", "build"))
    p.add_argument("--label")
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--protection", type=int, choices=(1, 10, 100), default=10)
    p.add_argument("--top", type=int, choices=range(1, 65), default=4)
    p.add_argument("--sites")
    args = p.parse_args()
    if args.command == "build":
        if not args.label or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in args.label):
            p.error("build requires an alphanumeric/hyphen label")
        build(args)
    else:
        {"prepare": prepare, "capture": capture, "control": control, "fit": fit}[args.command]()
