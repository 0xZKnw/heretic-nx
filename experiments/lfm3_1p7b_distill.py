"""Distil the existing NX teacher with benign-penalized input detectors."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import torch
from datasets import load_dataset
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from experiments.lfm3_1p7b import (
    DATASETS,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SHA256,
    SEED,
    batches,
    choose_candidate,
    compact_metrics,
    load_model,
    log_probs,
    mean_kl,
    publish_checkpoint,
    refusals,
    research_data,
    select_registry,
    validate_layout,
    write_json,
)
from heretic_nx.data.research_splits import build_research_split, verify_manifest_texts
from heretic_nx.hashing import sha256_file, sha256_json
from heretic_nx.optimize.ridge import conjugate_gradient

TEACHER_SHA256 = "5acd5ee0015891923a05ed814b6aecd76c56fbf2c572275458d083bd834a757f"
SAFE_COUNT = 1024
TARGET_COUNT = 128
TRAJECTORY_TOKENS = 32
TRAJECTORY_STRIDE = 4


def unload(model):
    model.to("cpu")
    gc.collect()
    torch.mps.empty_cache()


@torch.inference_mode()
def collect_inputs(model, tokenizer, prompts, sites, *, trajectory=False):
    storage = {site.id: [] for site in sites}
    state = {"step": -1, "alive": None}

    def begin(_module, inputs, kwargs):
        state["step"] += 1
        ids = kwargs.get("input_ids", inputs[0] if inputs else None)
        if ids is None:
            raise ValueError("input capture requires token IDs")
        if state["step"] == 0:
            state["alive"] = torch.ones(len(ids), dtype=torch.bool, device=ids.device)
        else:
            state["alive"] &= ids[:, -1] != tokenizer.eos_token_id

    handles = [model.register_forward_pre_hook(begin, with_kwargs=True)]
    for site in sites:

        def capture(_module, inputs, *, key=site.id):
            if not trajectory or state["step"] % TRAJECTORY_STRIDE == 0:
                value = (
                    inputs[0][state["alive"], -1]
                    .detach()
                    .to(device="cpu", dtype=torch.bfloat16)
                )
                if len(value):
                    storage[key].append(value)

        handles.append(
            model.get_submodule(site.module_path).register_forward_pre_hook(capture)
        )
    try:
        for index, batch in enumerate(batches(model, tokenizer, prompts, 4)):
            state["step"] = -1
            if trajectory:
                model.generate(
                    **batch,
                    do_sample=False,
                    max_new_tokens=TRAJECTORY_TOKENS,
                    pad_token_id=tokenizer.pad_token_id,
                )
            else:
                model(**batch, use_cache=False, logits_to_keep=1)
            if index % 16 == 0:
                print(
                    json.dumps(
                        {
                            "trajectory": trajectory,
                            "prompts": min((index + 1) * 4, len(prompts)),
                            "total": len(prompts),
                        }
                    ),
                    flush=True,
                )
    finally:
        for handle in handles:
            handle.remove()
    result = {key: torch.cat(values).contiguous() for key, values in storage.items()}
    if any(
        not torch.isfinite(value).all() or len(value) < 2 for value in result.values()
    ):
        raise ValueError("captured projection inputs must be finite nonempty matrices")
    return result


def geometry_data():
    result, manifests = {}, {}
    for label, count in (("safe", SAFE_COUNT), ("target", TARGET_COUNT)):
        dataset, revision = DATASETS[label]
        texts = [
            str(row["text"])
            for row in load_dataset(dataset, revision=revision, split="train")
        ]
        manifest = build_research_split(
            texts,
            purpose="geometry",
            dataset_id=dataset,
            revision=revision,
            source_split="train",
            seed=SEED,
            count=count,
        )
        result[label] = verify_manifest_texts(manifest, texts)
        manifests[label] = manifest.to_dict()
    return result, manifests


def fit_detector(harmful, safe, exact, safe_weight):
    if not 0 < safe_weight <= 1000:
        raise ValueError("safe weight must be in (0, 1000]")
    return conjugate_gradient(harmful, safe, exact, safe_lambda=safe_weight)


def cache_factors(factors, path, identity):
    # The same output axis is shared by penalty trials; safetensors requires independent storage.
    save_file(
        {key: value.clone().contiguous() for key, value in factors.items()},
        path,
        metadata={"identity": identity},
    )


@torch.no_grad()
def apply_factors(model, sites, originals, factors, weight, scale):
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("distilled scale must be finite and non-negative")
    for site in sites:
        key = f"lambda{weight:g}.{site.id}"
        axis, detector = factors[key + ".axis"], factors[key + ".detector"]
        original = originals[site.id].float()
        if axis.shape != (original.shape[0],) or detector.shape != (original.shape[1],):
            raise ValueError("distilled factor dimensions disagree")
        edited = original + scale * axis[:, None] @ detector[None, :]
        if edited.shape != original.shape or not torch.isfinite(edited).all():
            raise ValueError("distilled matrix is invalid")
        destination = model.get_submodule(site.module_path).weight
        rounded = edited.to(destination)
        if not torch.isfinite(rounded).all():
            raise ValueError("distilled matrix overflowed its deployment dtype")
        destination.copy_(rounded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-refusals", type=int, choices=range(25), default=2)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/lfm3-1p7b-distill"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if (
        sha256_file(args.source / "model.safetensors") != MODEL_SHA256
        or sha256_file(args.teacher / "model.safetensors") != TEACHER_SHA256
    ):
        raise ValueError("source or teacher checkpoint hash mismatch")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    prompts, manifests = geometry_data()
    identity = sha256_json(
        {
            "source": MODEL_SHA256,
            "teacher": TEACHER_SHA256,
            "manifests": manifests,
            "tokens": TRAJECTORY_TOKENS,
            "stride": TRAJECTORY_STRIDE,
            "torch": torch.__version__,
        }
    )
    model, tokenizer = load_model(args.source, "mps")
    registry = select_registry(validate_layout(model), 4, "all")
    sites = registry.sites
    originals = {
        site.id: model.get_submodule(site.module_path).weight.detach().cpu().clone()
        for site in sites
    }
    safe_path, target_path = (
        args.run_dir / "safe-inputs.safetensors",
        args.run_dir / "target-inputs.safetensors",
    )

    def cached(path):
        if not path.exists():
            return None
        with safe_open(path, framework="pt") as f:
            if f.metadata() != {"identity": identity}:
                raise ValueError("input cache provenance mismatch")
        return load_file(path)

    safe_inputs = cached(safe_path)
    if safe_inputs is None:
        safe_inputs = collect_inputs(model, tokenizer, prompts["safe"], sites)
        save_file(safe_inputs, safe_path, metadata={"identity": identity})
    target_inputs = cached(target_path)
    if target_inputs is None:
        unload(model)
        del model
        gc.collect()
        torch.mps.empty_cache()
        teacher, teacher_tokenizer = load_model(args.teacher, "mps")
        target_inputs = collect_inputs(
            teacher, teacher_tokenizer, prompts["target"], sites, trajectory=True
        )
        save_file(target_inputs, target_path, metadata={"identity": identity})
        unload(teacher)
        del teacher, teacher_tokenizer
        gc.collect()
        torch.mps.empty_cache()
        model, tokenizer = load_model(args.source, "mps")
    factors_path = args.run_dir / "factors.safetensors"
    diagnostics = []
    if factors_path.exists():
        with safe_open(factors_path, framework="pt") as f:
            if f.metadata() != {"identity": identity}:
                raise ValueError("factor cache provenance mismatch")
        factors = load_file(factors_path)
    else:
        factors = {}
        with safe_open(
            args.teacher / "model.safetensors", framework="pt"
        ) as teacher_weights:
            for index, site in enumerate(sites):
                delta = (
                    teacher_weights.get_tensor(site.module_path + ".weight").float()
                    - originals[site.id].float()
                )
                harmful = target_inputs[site.id].float()
                safe = safe_inputs[site.id].float()
                desired = harmful @ delta.T
                with torch.random.fork_rng():
                    torch.manual_seed(SEED + index)
                    _, _, vectors = torch.pca_lowrank(
                        desired, q=5, center=False, niter=2
                    )
                axis = vectors[:, 0].contiguous()
                exact = delta.T @ axis
                for weight in (1.0, 10.0, 100.0):
                    detector, iterations, residual = fit_detector(
                        harmful.to("mps"), safe.to("mps"), exact.to("mps"), weight
                    )
                    if not torch.isfinite(detector).all():
                        raise ValueError("nonfinite detector")
                    key = f"lambda{weight:g}.{site.id}"
                    factors[key + ".axis"], factors[key + ".detector"] = axis, detector
                    diagnostics.append(
                        {
                            "site": site.id,
                            "safe_weight": weight,
                            "cg_iterations": iterations,
                            "cg_residual_ratio": residual,
                            "target_relative_fit_error": float(
                                ((harmful @ detector) - (harmful @ exact)).norm()
                                / (harmful @ exact).norm().clamp_min(1e-12)
                            ),
                            "safe_to_target_rms": float(
                                (safe @ detector).square().mean().sqrt()
                                / (harmful @ exact)
                                .square()
                                .mean()
                                .sqrt()
                                .clamp_min(1e-12)
                            ),
                        }
                    )
                print(
                    json.dumps({"fitted_site": index + 1, "total": len(sites)}),
                    flush=True,
                )
        cache_factors(factors, factors_path, identity)
        write_json(args.run_dir / "fit.json", diagnostics)
    data, _ = research_data()
    base = log_probs(model, tokenizer, data["safe"]["selection"], 4)
    teacher_build = json.loads((args.teacher / "HERETIC_NX_BUILD.json").read_text())
    if (
        teacher_build["source_sha256"] != MODEL_SHA256
        or teacher_build["max_new_tokens"] != 96
        or teacher_build["enable_thinking"] is not False
    ):
        raise ValueError("teacher baseline protocol differs")
    baseline = teacher_build["baseline"]
    report = {
        "source_model": MODEL_ID,
        "revision": MODEL_REVISION,
        "source_sha256": MODEL_SHA256,
        "teacher_sha256": TEACHER_SHA256,
        "source_files": {
            p.name: sha256_file(p) for p in args.source.iterdir() if p.is_file()
        },
        "device": "mps",
        "dtype": "torch.bfloat16",
        "profile": "conditional-input-teacher-distillation",
        "row_norms_restored": False,
        "kl_cap": 0.05,
        "max_selection_refusals": args.max_refusals,
        "max_new_tokens": 96,
        "enable_thinking": False,
        "inputs_identity": identity,
        "safe_inputs_sha256": sha256_file(safe_path),
        "target_inputs_sha256": sha256_file(target_path),
        "factors_sha256": sha256_file(factors_path),
        "manifests": manifests,
        "baseline": baseline,
        "candidates": [],
    }
    for weight in (1.0, 10.0, 100.0):
        for scale in (1.0, 1.5, 2.0):
            apply_factors(model, sites, originals, factors, weight, scale)
            kl = mean_kl(
                base, log_probs(model, tokenizer, data["safe"]["selection"], 4)
            )
            row = {
                "safe_weight": weight,
                "scale": scale,
                "kl": kl,
                "complete": False,
                "marker_count": None,
            }
            if kl <= 0.05:
                screen = refusals(
                    model, tokenizer, data["target"]["selection"], 4, max_new_tokens=16
                )
                row["screen"] = compact_metrics(screen)
                # Prefix marker hits are a lower bound, never a complete acceptance report.
                if (
                    screen["marker_count"] <= args.max_refusals
                    and screen["empty_responses"] == 0
                ):
                    row.update(
                        refusals(model, tokenizer, data["target"]["selection"], 4)
                    )
                    row["complete"] = True
                else:
                    row["skipped_target"] = "prefix marker bound failed"
            else:
                row["skipped_target"] = "KL cap failed"
            report["candidates"].append(row)
            write_json(args.run_dir / "search.json", report)
            print(
                json.dumps(
                    {k: row[k] for k in ("safe_weight", "scale", "kl", "marker_count")}
                ),
                flush=True,
            )
    selected = choose_candidate(
        report["candidates"],
        baseline["marker_count"],
        kl_cap=0.05,
        max_refusals=args.max_refusals,
    )
    apply_factors(
        model, sites, originals, factors, selected["safe_weight"], selected["scale"]
    )
    report["selected"] = {
        k: selected[k] for k in ("safe_weight", "scale", "kl", "marker_count")
    }
    publish_checkpoint(args, model, tokenizer, report)


if __name__ == "__main__":
    main()
