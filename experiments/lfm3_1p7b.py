#!/usr/bin/env python3
"""Pinned LFM3 BF16 research build using NX's existing residual-stream editor.

Search uses group-disjoint training partitions. The dataset test split and
paired capability slice are opened only by the separate validate command.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from heretic_nx.data.research_splits import build_research_split, verify_manifest_texts
from heretic_nx.edits.activation_op import ActivationOperator
from heretic_nx.edits.residual_stream import (
    apply_residual_stream_weight_edits,
    build_residual_stream_weight_editors,
    snapshot_residual_stream_weights,
)
from heretic_nx.eval.capability import paired_bootstrap_interval
from heretic_nx.geometry.residual import (
    fit_residual_stream_axes,
    protect_residual_stream_axes,
)
from heretic_nx.hashing import canonical_json, sha256_file, sha256_json
from heretic_nx.model import discover_semantic_sites

MODEL_ID = "LiquidAI/LFM3-1.7B"
MODEL_REVISION = "1da06bec6a819da10b54c67e5c90972b46d81871"
MODEL_SHA256 = "135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330"
DATASETS = {
    "safe": ("mlabonne/harmless_alpaca", "02c6a92cfcf11bb0c387334f8146d149d65b587f"),
    "target": (
        "mlabonne/harmful_behaviors",
        "01cead01398926d81f7c52bdb790ee8cf77ebba7",
    ),
}
SEED = 1703
GEOMETRY_COUNT = 128
SELECTION_COUNT = 24
MAX_NEW_TOKENS = 96
KL_CAP = 0.03
FIRST_EDIT_LAYER = 4


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value) + b"\n")


def compact_metrics(value):
    """Ship metrics and response hashes; retain raw generations only in local runs."""
    if isinstance(value, dict):
        return {
            ("responses_sha256" if key == "responses" else key): (
                sha256_json(item) if key == "responses" else compact_metrics(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [compact_metrics(item) for item in value]
    return value


def validate_layout(model):
    """Reject missing branches and architectures other than the pinned LFM3."""
    config = model.config
    if config.model_type != "lfm3" or config.hidden_size != 2048:
        raise ValueError("expected LFM3 with residual width 2048")
    types = tuple(config.layer_types)
    if len(types) != 24 or sorted(types).count("conv") != 13:
        raise ValueError("expected 24 LFM3 blocks including 13 convolutions")
    if types.count("full_attention") != 6 or types.count("sliding_attention") != 5:
        raise ValueError("expected six global and five sliding attention blocks")
    registry = discover_semantic_sites(model)
    for layer, kind in enumerate(types):
        sites = [s for s in registry.sites if s.layer == layer and s.family != "block"]
        expected = {"ffn", "liv" if kind == "conv" else "gqa"}
        if len(sites) != 2 or {s.family for s in sites} != expected:
            raise ValueError(f"incomplete LFM3 projection layout at layer {layer}")
        if any(s.output_dim != 2048 or s.stream_dim != 2048 for s in sites):
            raise ValueError(f"invalid LFM3 output width at layer {layer}")
    return registry


def render(tokenizer, prompts):
    return tokenizer.apply_chat_template(
        [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            for prompt in prompts
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def batches(model, tokenizer, prompts, batch_size):
    for start in range(0, len(prompts), batch_size):
        # Fail rather than silently truncate a calibration or evaluation row.
        batch = tokenizer(
            render(tokenizer, prompts[start : start + batch_size]),
            padding=True,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        if (
            batch.input_ids.shape[1]
            > model.config.max_position_embeddings - MAX_NEW_TOKENS
        ):
            raise ValueError("prompt exceeds the model context")
        yield batch.to(model.device)


@torch.inference_mode()
def collect(model, tokenizer, prompts, batch_size):
    states = []
    handles = [
        layer.register_forward_hook(
            lambda _module, _inputs, output: states.append(output[:, -1].float().cpu())
        )
        for layer in model.model.layers
    ]
    rows = []
    try:
        for batch in batches(model, tokenizer, prompts, batch_size):
            states.clear()
            model(**batch, use_cache=False, logits_to_keep=1)
            rows.append(torch.stack(states, dim=1))
            print(
                json.dumps(
                    {"collected": sum(len(row) for row in rows), "total": len(prompts)}
                ),
                flush=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    return torch.cat(rows)


@torch.inference_mode()
def log_probs(model, tokenizer, prompts, batch_size):
    return torch.cat(
        [
            model(**batch, use_cache=False, logits_to_keep=1)
            .logits[:, -1]
            .float()
            .log_softmax(-1)
            .cpu()
            for batch in batches(model, tokenizer, prompts, batch_size)
        ]
    )


def mean_kl(base, candidate):
    if base.shape != candidate.shape or base.ndim != 2 or not len(base):
        raise ValueError("KL requires aligned nonempty log-probability matrices")
    if not torch.isfinite(base).all() or not torch.isfinite(candidate).all():
        raise ValueError("KL inputs must be finite")
    return float((base.exp() * (base - candidate)).sum(-1).mean())


@torch.inference_mode()
def refusals(model, tokenizer, prompts, batch_size):
    from experiments.lfm25_2p6b_residual_stream import (
        REFUSAL_MARKERS,
        normalized_response,
    )

    responses = []
    empty_responses = 0
    started = time.monotonic()
    for batch in batches(model, tokenizer, prompts, batch_size):
        output = model.generate(
            **batch,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
        )
        responses.extend(
            tokenizer.batch_decode(
                output[:, batch.input_ids.shape[1] :], skip_special_tokens=False
            )
        )
        empty_responses += sum(
            not text.strip()
            for text in tokenizer.batch_decode(
                output[:, batch.input_ids.shape[1] :], skip_special_tokens=True
            )
        )
        print(
            json.dumps({"generated": len(responses), "total": len(prompts)}), flush=True
        )
    hits = [
        int(any(marker in normalized_response(text) for marker in REFUSAL_MARKERS))
        for text in responses
    ]
    return {
        "count": len(hits),
        "marker_count": sum(hits),
        "marker_hits": hits,
        "responses": responses,
        "empty_responses": empty_responses,
        "seconds": time.monotonic() - started,
    }


def load_model(path, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, local_files_only=True
    )
    tokenizer.padding_side = "left"
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model = (
        AutoModelForCausalLM.from_pretrained(
            path,
            trust_remote_code=True,
            local_files_only=True,
            dtype=dtype,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
    )
    validate_layout(model)
    return model, tokenizer


def research_data():
    from datasets import load_dataset

    data, manifests = {}, {}
    for label, (dataset_id, revision) in DATASETS.items():
        texts = [
            str(row["text"])
            for row in load_dataset(dataset_id, revision=revision, split="train")
        ]
        data[label] = {}
        for purpose, count in (
            ("geometry", GEOMETRY_COUNT),
            ("selection", SELECTION_COUNT),
        ):
            manifest = build_research_split(
                texts,
                purpose=purpose,
                dataset_id=dataset_id,
                revision=revision,
                source_split="train",
                seed=SEED,
                count=count,
            )
            data[label][purpose] = verify_manifest_texts(manifest, texts)
            manifests[f"{label}:{purpose}"] = manifest.to_dict()
    return data, manifests


def choose_candidate(candidates, base_markers, *, kl_cap=KL_CAP, max_refusals=None):
    """Minimize drift once the requested refusal target is reached."""
    eligible = [
        row
        for row in candidates
        if row["marker_count"] < base_markers
        and 0 <= row["kl"] <= kl_cap
        and (max_refusals is None or row["marker_count"] <= max_refusals)
        and row.get("empty_responses", 0) == 0
    ]
    if not eligible:
        raise RuntimeError(
            "no candidate improves the refusal proxy within the KL budget"
        )
    return min(
        eligible,
        key=lambda row: (
            (row["marker_count"], row["kl"])
            if max_refusals is None
            else (row["kl"], row["marker_count"])
        ),
    )


def fit_profiles(safe, target):
    # The first convolution sees only the fixed chat suffix and has zero contrast.
    safe, target = safe[:, FIRST_EDIT_LAYER:], target[:, FIRST_EDIT_LAYER:]
    raw = fit_residual_stream_axes(safe, target, folds=3)
    protected = protect_residual_stream_axes(
        safe, target, raw, capability_rank=4, seed=SEED
    )
    return {
        name: dict(enumerate(axes, start=FIRST_EDIT_LAYER))
        for name, axes in (
            ("raw", raw),
            ("protected", tuple(axis.evidence for axis in protected)),
        )
    }


def sandwich_weight_operator(axis, gain, row_norms):
    """Express a residual direction in the editor's row-normalized coordinates."""
    axis, gain, row_norms = (
        value.detach().float().cpu() for value in (axis, gain, row_norms)
    )
    if axis.ndim != 1 or gain.shape != axis.shape or row_norms.shape != axis.shape:
        raise ValueError("axis, sandwich gain and row norms must be aligned vectors")
    coordinates = gain * row_norms
    if (
        not torch.isfinite(axis).all()
        or not torch.isfinite(coordinates).all()
        or bool((row_norms <= 0).any())
        or bool((coordinates.abs() <= 1e-7).any())
    ):
        raise ValueError("sandwich coordinates must be finite and invertible")
    return ActivationOperator(
        (axis / coordinates)[:, None], (axis * coordinates)[:, None], 1.0
    )


def bind_profile(
    model, registry, axes, originals, *, shared_axis=None, sandwich_aware=False
):
    if shared_axis is not None:
        axes = {layer: axes[shared_axis] for layer in axes}
    editors = build_residual_stream_weight_editors(registry, axes)
    if not sandwich_aware:
        return editors
    result = []
    for editor in editors:
        layer = model.model.layers[editor.site.layer]
        norm = (
            layer.post_ffn_norm
            if editor.site.family == "ffn"
            else layer.post_operator_norm
        )
        row_norms = originals[editor.site_id].float().norm(dim=1)
        operator = sandwich_weight_operator(
            editor.evidence.axis, norm.weight, row_norms
        )
        result.append(replace(editor, operator=operator))
    return tuple(result)


def select_registry(registry, first_layer, family):
    families = {
        "all": {"ffn", "gqa", "liv"},
        "ffn": {"ffn"},
        "operator": {"gqa", "liv"},
    }[family]
    return replace(
        registry,
        sites=tuple(
            site
            for site in registry.sites
            if site.layer >= first_layer and site.family in families
        ),
    )


def search(args):
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if sha256_file(args.source / "model.safetensors") != MODEL_SHA256:
        raise ValueError("source checkpoint differs from the pinned official BF16")
    data, manifests = research_data()
    write_json(args.run_dir / "splits.json", manifests)
    model, tokenizer = load_model(args.source, args.device)
    registry = validate_layout(model)
    registry = select_registry(registry, args.first_edit_layer, args.site_family)
    cache = args.run_dir / "residuals.safetensors"
    identity = sha256_json(
        {
            "model": MODEL_SHA256,
            "splits": manifests,
            "template": sha256_file(args.source / "chat_template.jinja"),
            "device": args.device,
            "torch": torch.__version__,
            "dtype": str(model.dtype),
            "batch_size": args.batch_size,
            "capture": "raw-last-token-per-block-closed-thinking-v1",
        }
    )
    from safetensors import safe_open

    if cache.exists():
        with safe_open(cache, framework="pt") as handle:
            if handle.metadata() != {"identity": identity}:
                raise ValueError("residual cache provenance mismatch")
        states = load_file(cache)
    else:
        states = {
            label: collect(model, tokenizer, data[label]["geometry"], args.batch_size)
            for label in DATASETS
        }
        save_file(states, cache, metadata={"identity": identity})
    profiles = fit_profiles(states["safe"], states["target"])
    editors = build_residual_stream_weight_editors(registry, profiles["raw"])
    originals = snapshot_residual_stream_weights(model, editors)

    def bind(axes):
        return bind_profile(
            model,
            registry,
            axes,
            originals,
            shared_axis=args.shared_axis,
            sandwich_aware=args.sandwich_aware,
        )

    base_logp = log_probs(model, tokenizer, data["safe"]["selection"], args.batch_size)
    baseline = refusals(model, tokenizer, data["target"]["selection"], args.batch_size)
    report = {
        "source_model": MODEL_ID,
        "revision": MODEL_REVISION,
        "source_sha256": MODEL_SHA256,
        "source_files": {
            p.name: sha256_file(p) for p in args.source.iterdir() if p.is_file()
        },
        "device": args.device,
        "dtype": str(model.dtype),
        "max_new_tokens": MAX_NEW_TOKENS,
        "enable_thinking": False,
        "kl_cap": args.kl_cap,
        "max_selection_refusals": args.max_refusals,
        "shared_axis_layer": args.shared_axis,
        "sandwich_aware": args.sandwich_aware,
        "first_edit_layer": args.first_edit_layer,
        "site_family": args.site_family,
        "active_sites": len(editors),
        "baseline": baseline,
        "candidates": [],
        "profile_limit": "LFM3 sandwich normalization makes residual edits an empirical intervention, not an exact residual projector.",
        "claim_limit": "Lexical refusal proxy only; no semantic judge or capability certification.",
    }
    for profile, axes in profiles.items():
        editors = bind(axes)
        for scale in args.scales:
            strengths = {e.site_id: scale for e in editors}
            apply_residual_stream_weight_edits(model, editors, originals, strengths)
            target = refusals(
                model, tokenizer, data["target"]["selection"], args.batch_size
            )
            kl = mean_kl(
                base_logp,
                log_probs(model, tokenizer, data["safe"]["selection"], args.batch_size),
            )
            row = {"profile": profile, "scale": scale, "kl": kl, **target}
            report["candidates"].append(row)
            write_json(args.run_dir / "search.json", report)
            print(
                json.dumps(
                    {k: row[k] for k in ("profile", "scale", "kl", "marker_count")}
                ),
                flush=True,
            )
    selected = choose_candidate(
        report["candidates"],
        baseline["marker_count"],
        kl_cap=args.kl_cap,
        max_refusals=args.max_refusals,
    )
    editors = bind(profiles[selected["profile"]])
    apply_residual_stream_weight_edits(
        model, editors, originals, {e.site_id: selected["scale"] for e in editors}
    )
    report["selected"] = {
        k: selected[k] for k in ("profile", "scale", "kl", "marker_count")
    }
    # Publish a complete local research checkpoint, preserving the official custom code and license.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import tempfile

    with tempfile.TemporaryDirectory(
        dir=args.output.parent, prefix=".lfm3-build-"
    ) as temp:
        staging = Path(temp)
        model.to("cpu").to(torch.bfloat16).save_pretrained(
            staging, safe_serialization=True, max_shard_size="5GB"
        )
        tokenizer.save_pretrained(staging)
        for path in args.source.iterdir():
            if path.is_file() and (
                path.suffix == ".py" or path.name in ("LICENSE", "chat_template.jinja")
            ):
                shutil.copy2(path, staging / path.name)
        report["output_sha256"] = sha256_file(staging / "model.safetensors")
        write_json(staging / "HERETIC_NX_BUILD.json", compact_metrics(report))
        staging.rename(args.output)
    write_json(args.run_dir / "search.json", report)
    print(
        json.dumps({"output": str(args.output), "selected": report["selected"]}),
        flush=True,
    )


@torch.inference_mode()
def capability(model, tokenizer, rows, batch_size):
    letters = [tokenizer.encode(letter, add_special_tokens=False) for letter in "ABCD"]
    if any(len(ids) != 1 for ids in letters):
        raise ValueError("MCQ answer letters must each be a single token")
    ids = torch.tensor([item[0] for item in letters], device=model.device)
    predictions = []
    for batch in batches(model, tokenizer, [row["prompt"] for row in rows], batch_size):
        logits = model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1]
        predictions.extend(logits.index_select(-1, ids).argmax(-1).cpu().tolist())
    correct = [
        int(prediction == row["answer"]) for prediction, row in zip(predictions, rows)
    ]
    return {
        "count": len(rows),
        "accuracy": sum(correct) / len(correct),
        "predictions": predictions,
        "correctness": correct,
    }


def validate(args):
    from datasets import load_dataset

    from experiments.lfm25_residual_stream_capability import selected_rows

    build = json.loads((args.output / "HERETIC_NX_BUILD.json").read_text())
    if sha256_file(args.output / "model.safetensors") != build["output_sha256"]:
        raise ValueError("candidate weights changed after selection")
    for filename, digest in build["source_files"].items():
        if sha256_file(args.source / filename) != digest:
            raise ValueError(f"source file changed after selection: {filename}")
    tests = {
        label: [
            str(row["text"])
            for row in load_dataset(dataset_id, revision=revision, split="test[:104]")
        ]
        for label, (dataset_id, revision) in DATASETS.items()
    }
    rows = selected_rows()
    report = {
        "source_sha256": MODEL_SHA256,
        "output_sha256": build["output_sha256"],
        "test_rows_sha256": sha256_json(tests),
        "capability_rows_sha256": sha256_json(rows),
        "test_source_split": "test[:104]",
        "selection_kl_cap": build["kl_cap"],
        "max_new_tokens": MAX_NEW_TOKENS,
        "enable_thinking": False,
        "arms": {},
    }
    baseline_logp = None
    for arm, path in (("base", args.source), ("heretic_nx", args.output)):
        model, tokenizer = load_model(path, args.device)
        arm_report = {
            "target": refusals(model, tokenizer, tests["target"], args.batch_size),
            "safe": refusals(model, tokenizer, tests["safe"], args.batch_size),
            "capability": capability(model, tokenizer, rows, args.batch_size),
        }
        current_logp = log_probs(model, tokenizer, tests["safe"], args.batch_size)
        if baseline_logp is None:
            baseline_logp = current_logp
        else:
            report["first_token_kl"] = mean_kl(baseline_logp, current_logp)
        report["arms"][arm] = arm_report
        write_json(args.run_dir / "validation.json", report)
        del model
        import gc

        gc.collect()
        if args.device == "mps":
            torch.mps.empty_cache()
    base, edited = report["arms"]["base"], report["arms"]["heretic_nx"]
    report["capability_interval"] = asdict(
        paired_bootstrap_interval(
            base["capability"]["correctness"],
            edited["capability"]["correctness"],
            margin=0.03,
            seed=SEED,
        )
    )
    report["proxy_gates"] = {
        "target_improved": edited["target"]["marker_count"]
        < base["target"]["marker_count"],
        "target_near_zero": edited["target"]["marker_count"] <= 4,
        "safe_not_worse": edited["safe"]["marker_count"]
        <= base["safe"]["marker_count"],
        "kl_within_budget": 0 <= report["first_token_kl"] <= build["kl_cap"],
        "capability_noninferior": report["capability_interval"][
            "noninferiority_passed"
        ],
    }
    report["claim_limit"] = (
        "Research artifact; lexical proxies and a narrow first-token MCQ slice, without semantic certification."
    )
    write_json(args.run_dir / "validation.json", report)
    write_json(args.output / "HERETIC_NX_VALIDATION.json", compact_metrics(report))
    print(
        json.dumps(
            {
                "gates": report["proxy_gates"],
                "kl": report["first_token_kl"],
                "capability_interval": report["capability_interval"],
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("search", "validate"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/lfm3-1p7b"))
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="mps")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--scales", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    parser.add_argument("--kl-cap", type=float, default=KL_CAP)
    parser.add_argument("--max-refusals", type=int)
    parser.add_argument("--shared-axis", type=int, choices=range(FIRST_EDIT_LAYER, 24))
    parser.add_argument("--sandwich-aware", action="store_true")
    parser.add_argument(
        "--first-edit-layer",
        type=int,
        choices=range(FIRST_EDIT_LAYER, 24),
        default=FIRST_EDIT_LAYER,
    )
    parser.add_argument(
        "--site-family", choices=("all", "ffn", "operator"), default="all"
    )
    args = parser.parse_args()
    if args.batch_size < 1 or any(not 0 < value <= 4 for value in args.scales):
        parser.error("batch size must be positive and scales in (0, 4]")
    if not 0 <= args.kl_cap <= 10 or (
        args.max_refusals is not None and not 0 <= args.max_refusals <= SELECTION_COUNT
    ):
        parser.error("KL cap must be in [0, 10] and refusal target in [0, 24]")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    {"search": search, "validate": validate}[args.command](args)


if __name__ == "__main__":
    main()
