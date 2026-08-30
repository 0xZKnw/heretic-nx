from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import heretic_nx.eval.native_logits as native_logits_module
from heretic_nx.eval.native_logits import (
    RAW_LOGIT_FORMAT,
    attest_tokenizer_assets,
    collect_native_raw_logits,
    require_tokenizer_identity,
)
from heretic_nx.hashing import sha256_bytes, sha256_file, sha256_json


SCHEMA = "test-native-raw-logits-v1"


FAKE_COLLECTOR = r'''#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import re
import struct
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("tokens")
parser.add_argument("output")
parser.add_argument("--n-ctx", type=int)
parser.add_argument("--n-batch", type=int)
parser.add_argument("--n-ubatch", type=int)
parser.add_argument("--threads", type=int, required=True)
parser.add_argument("--gpu-layers", type=int, required=True)
parser.add_argument("--expected-count", type=int, required=True)
parser.add_argument("--expected-vocab", type=int, required=True)
parser.add_argument("--backend-dir", required=True)
args = parser.parse_args()
split = re.fullmatch(r"(.+)-00001-of-([0-9]{5})[.]gguf", Path(args.model).name)
if split:
    total = int(split.group(2))
    for index in range(1, total + 1):
        sibling = Path(args.model).with_name(
            f"{split.group(1)}-{index:05d}-of-{total:05d}.gguf"
        )
        if not sibling.is_file():
            sys.exit(8)
rows = [line.split() for line in open(args.tokens, encoding="ascii")]
if len(rows) != args.expected_count:
    sys.exit(7)
maximum = max(map(len, rows))
with open(args.output, "xb") as stream:
    for row in range(len(rows)):
        values = [float(row + token + 1) for token in range(args.expected_vocab)]
        stream.write(struct.pack("<" + "f" * len(values), *values))
summary = {
    "schema_version": "llama-raw-logits-native-v2",
    "count": len(rows),
    "vocab_size": args.expected_vocab,
    "values_per_row": args.expected_vocab,
    "maximum_prompt_tokens": maximum,
    "n_ctx": args.n_ctx or max(32, maximum),
    "n_batch": args.n_batch or max(32, maximum),
    "n_ubatch": args.n_ubatch or args.n_batch or max(32, maximum),
    "threads": args.threads,
    "gpu_layers": args.gpu_layers,
    "gpu_offload_supported": True,
    "model_context_train": 4096,
    "model_size_bytes": 1024,
    "model_parameter_count": 100,
    "model_description": "fake Q8_0",
    "system_info": "fake-runtime",
    "module_inventory_method": (
        "windows-toolhelp32" if sys.platform == "win32" else
        "macos-dyld-images" if sys.platform == "darwin" else
        "linux-proc-self-maps"
    ),
    "loaded_modules": [
        str(Path(args.output).parent / Path(sys.argv[0]).name),
        str(Path(args.backend_dir) / "libggml.so"),
        str(Path(args.backend_dir) / "libggml-metal.so"),
        str(Path(args.backend_dir) / "libllama.so"),
        str(Path(sys.executable).resolve()),
    ],
    "float_format": "float32-little-endian",
    "memory_cleared_between_rows": True,
    "backend_loading": "explicit-directory",
    "model_load_seconds": 0.01,
    "decode_seconds": 0.02,
    "total_seconds": 0.03,
}
print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
'''


def test_native_logits_api_is_publicly_exported() -> None:
    from heretic_nx import eval as eval_api

    assert eval_api.attest_tokenizer_assets is attest_tokenizer_assets
    assert eval_api.collect_native_raw_logits is collect_native_raw_logits
    assert eval_api.require_tokenizer_identity is require_tokenizer_identity


def test_tokenizer_attestation_rejects_asset_added_during_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer_root = tmp_path / "tokenizer"
    tokenizer_root.mkdir()
    (tokenizer_root / "tokenizer.json").write_text("{}", encoding="utf-8")
    identity = attest_tokenizer_assets(
        tokenizer_root,
        vocab_size=3,
        tokenizer_class="FakeTokenizer",
    )
    real_hash = native_logits_module.sha256_file
    added = False

    def racing_hash(path: Path) -> str:
        nonlocal added
        result = real_hash(path)
        if not added:
            added = True
            (tokenizer_root / "chat_template.jinja").write_text(
                "{{ messages }}", encoding="utf-8"
            )
        return result

    monkeypatch.setattr(native_logits_module, "sha256_file", racing_hash)

    with pytest.raises(RuntimeError, match="asset set changed while re-hashing"):
        require_tokenizer_identity(identity)


def _fixture(tmp_path: Path) -> dict[str, object]:
    tokenizer_root = tmp_path / "tokenizer"
    tokenizer_root.mkdir()
    (tokenizer_root / "tokenizer.json").write_text("{}", encoding="utf-8")
    tokenizer = attest_tokenizer_assets(
        tokenizer_root,
        vocab_size=3,
        tokenizer_class="FakeTokenizer",
    )
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fake-immutable-model")
    executable = tmp_path / "fake-collector"
    executable.write_text(FAKE_COLLECTOR, encoding="utf-8")
    executable.chmod(0o700)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "libggml.so").write_bytes(b"fake-ggml-runtime")
    (runtime / "libggml-metal.so").write_bytes(b"fake-metal-runtime")
    (runtime / "libllama.so").write_bytes(b"fake-llama-runtime")
    (runtime / "default.metallib").write_bytes(b"fake-metal-kernels")
    return {
        "token_rows": [[0, 1], [2]],
        "tokenizer_identity": tokenizer,
        "model_path": model,
        "output_path": tmp_path / "logits.raw.bin",
        "schema_version": SCHEMA,
        "label": "candidate",
        "model_alias": "candidate.gguf",
        "executable_path": executable,
        "runtime_library_dirs": [runtime],
    }


def test_native_raw_logits_collects_attested_no_clobber_artifact(
    tmp_path: Path,
) -> None:
    arguments = _fixture(tmp_path)

    result = collect_native_raw_logits(**arguments)

    assert result.reused is False
    assert result.data_path.read_bytes() == np.array(
        [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=np.float32
    ).tobytes()
    progress = result.progress
    serialized = b"0 1\n2\n"
    assert progress["prompt_tokens_sha256"] == sha256_json([[0, 1], [2]])
    assert progress["token_file_sha256"] == sha256_bytes(serialized)
    assert progress["artifact_sha256"] == sha256_file(arguments["model_path"])
    assert progress["artifact_files"] == progress["runtime_model"]["artifact_files"]
    assert progress["artifact_files"][0]["filename"] == "model.gguf"
    assert progress["data_sha256"] == sha256_file(result.data_path)
    assert progress["data_format"] == RAW_LOGIT_FORMAT
    assert progress["completed"] == 2
    assert progress["runtime_model"]["endpoint"] == "native://llama-raw-logits"
    assert progress["runtime_protocol"]["memory_cleared_between_rows"] is True
    pinned_modules = [
        module
        for module in progress["runtime_protocol"]["loaded_modules"]
        if module["scope"] == "pinned-runtime"
    ]
    assert {module["filename"] for module in pinned_modules} == {
        "libggml.so",
        "libggml-metal.so",
        "libllama.so",
    }
    assert {
        record["filename"]
        for record in progress["collector"]["runtime_libraries"]
    } == {
        "default.metallib",
        "libggml.so",
        "libggml-metal.so",
        "libllama.so",
    }
    assert progress["runtime_protocol"]["gpu_offload_supported"] is True


def test_native_raw_logits_fails_closed_without_requested_gpu_offload(
    tmp_path: Path,
) -> None:
    arguments = _fixture(tmp_path)
    executable = Path(arguments["executable_path"])
    source = executable.read_text(encoding="utf-8").replace(
        '"gpu_offload_supported": True',
        '"gpu_offload_supported": False',
    )
    executable.write_text(source, encoding="utf-8")

    with pytest.raises(RuntimeError, match="lacks requested GPU layer offload"):
        collect_native_raw_logits(**arguments)

    assert not Path(arguments["output_path"]).exists()

    result = collect_native_raw_logits(**arguments, gpu_layers=0)
    assert result.progress["runtime_protocol"]["gpu_layers"] == 0
    assert result.progress["runtime_protocol"]["gpu_offload_supported"] is False


def test_native_raw_logits_reuses_only_exact_immutable_artifact(
    tmp_path: Path,
) -> None:
    arguments = _fixture(tmp_path)
    first = collect_native_raw_logits(**arguments)

    second = collect_native_raw_logits(**arguments)

    assert first.data_path == second.data_path
    assert second.reused is True
    assert second.progress == first.progress


def test_native_raw_logits_rejects_partial_destination(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    Path(arguments["output_path"]).write_bytes(b"do-not-overwrite")

    with pytest.raises(RuntimeError, match="partial.*refusing to overwrite"):
        collect_native_raw_logits(**arguments)

    assert Path(arguments["output_path"]).read_bytes() == b"do-not-overwrite"


def test_native_raw_logits_rejects_changed_model_or_runtime(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    collect_native_raw_logits(**arguments)
    Path(arguments["model_path"]).write_bytes(b"changed-model")

    with pytest.raises(RuntimeError, match="stale.*model artifact"):
        collect_native_raw_logits(**arguments)

    Path(arguments["model_path"]).write_bytes(b"fake-immutable-model")
    runtime = Path(arguments["runtime_library_dirs"][0]) / "libggml.so"
    runtime.write_bytes(b"changed-runtime")
    with pytest.raises(RuntimeError, match="stale.*runtime bundle"):
        collect_native_raw_logits(**arguments)


def test_native_raw_logits_failure_does_not_publish_partial_files(
    tmp_path: Path,
) -> None:
    arguments = _fixture(tmp_path)
    executable = Path(arguments["executable_path"])
    executable.write_text("#!/bin/sh\nexit 9\n", encoding="utf-8")
    executable.chmod(0o700)

    with pytest.raises(RuntimeError, match="exit code 9"):
        collect_native_raw_logits(**arguments)

    output = Path(arguments["output_path"])
    assert not output.exists()
    assert not output.with_suffix(".progress.json").exists()


def test_native_raw_logits_rejects_runtime_loaded_outside_pinned_bundle(
    tmp_path: Path,
) -> None:
    arguments = _fixture(tmp_path)
    escaped = tmp_path / "escaped" / "libllama.so"
    escaped.parent.mkdir()
    escaped.write_bytes(b"unattested-llama-runtime")
    executable = Path(arguments["executable_path"])
    source = executable.read_text(encoding="utf-8").replace(
        'str(Path(args.backend_dir) / "libllama.so")',
        'str(Path(args.backend_dir).parent / "escaped" / "libllama.so")',
    )
    executable.write_text(source, encoding="utf-8")

    with pytest.raises(RuntimeError, match="escaped the pinned runtime directory"):
        collect_native_raw_logits(**arguments)

    assert not Path(arguments["output_path"]).exists()


def test_native_raw_logits_cleanup_never_unlinks_foreign_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _fixture(tmp_path)
    output = Path(arguments["output_path"])
    progress = output.with_suffix(".progress.json")
    real_link = native_logits_module.os.link

    def racing_link(source, destination, *args, **kwargs):
        if Path(destination) == progress:
            output.unlink()
            output.write_bytes(b"foreign-race-winner")
            raise FileExistsError("simulated progress publication race")
        return real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(native_logits_module.os, "link", racing_link)

    with pytest.raises(FileExistsError, match="publication race"):
        collect_native_raw_logits(**arguments)

    assert output.read_bytes() == b"foreign-race-winner"
    assert not progress.exists()


def test_native_raw_logits_validates_context_tokens_and_tokenizer(
    tmp_path: Path,
) -> None:
    arguments = _fixture(tmp_path)
    with pytest.raises(ValueError, match="context_size"):
        collect_native_raw_logits(**arguments, context_size=1)
    with pytest.raises(ValueError, match="ubatch_size"):
        collect_native_raw_logits(**arguments, batch_size=2, ubatch_size=3)
    with pytest.raises(ValueError, match="uint32"):
        collect_native_raw_logits(**arguments, context_size=2**32)
    with pytest.raises(ValueError, match="threads.*int32"):
        collect_native_raw_logits(**arguments, threads=2**31)
    with pytest.raises(ValueError, match="gpu_layers.*int32"):
        collect_native_raw_logits(**arguments, gpu_layers=2**31)
    with pytest.raises(ValueError, match="invalid token id"):
        collect_native_raw_logits(**{**arguments, "token_rows": [[3]]})

    tokenizer = arguments["tokenizer_identity"]
    tokenizer_path = Path(tokenizer["root"]) / "tokenizer.json"
    tokenizer_path.write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="tokenizer asset bytes changed"):
        require_tokenizer_identity(tokenizer)


def test_tokenizer_attestation_rejects_path_escape(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    identity = dict(arguments["tokenizer_identity"])
    files = [dict(identity["files"][0])]
    files[0]["relative_path"] = "../model.gguf"
    identity["files"] = files

    with pytest.raises(RuntimeError, match="relative path"):
        require_tokenizer_identity(identity)


def test_tokenizer_attestation_rejects_added_or_removed_assets(
    tmp_path: Path,
) -> None:
    arguments = _fixture(tmp_path)
    identity = arguments["tokenizer_identity"]
    tokenizer_root = Path(identity["root"])
    added = tokenizer_root / "chat_template.jinja"
    added.write_text("{{ messages }}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="asset set changed"):
        require_tokenizer_identity(identity)

    added.unlink()
    (tokenizer_root / "tokenizer.json").unlink()
    with pytest.raises(RuntimeError, match="asset set changed"):
        require_tokenizer_identity(identity)


def test_native_raw_logits_attests_complete_split_gguf_bundle(
    tmp_path: Path,
) -> None:
    arguments = _fixture(tmp_path)
    Path(arguments["model_path"]).unlink()
    first = tmp_path / "model-00001-of-00002.gguf"
    second = tmp_path / "model-00002-of-00002.gguf"
    first.write_bytes(b"split-model-first")
    second.write_bytes(b"split-model-second")
    arguments["model_path"] = first

    result = collect_native_raw_logits(**arguments)

    files = result.progress["artifact_files"]
    assert [record["filename"] for record in files] == [first.name, second.name]
    assert result.progress["runtime_model"]["artifact_size_bytes"] == (
        first.stat().st_size + second.stat().st_size
    )
    expected = sha256_json(
        {
            "schema_version": "heretic-nx-gguf-artifact-bundle-v1",
            "files": files,
        }
    )
    assert result.progress["artifact_sha256"] == expected


@pytest.mark.parametrize(
    "filename",
    ("model-00001-of-00001.gguf", "model-00001-of-00002.GGUF"),
)
def test_native_raw_logits_treats_noncanonical_split_names_as_simple_models(
    tmp_path: Path,
    filename: str,
) -> None:
    arguments = _fixture(tmp_path)
    original = Path(arguments["model_path"])
    renamed = original.with_name(filename)
    original.rename(renamed)
    arguments["model_path"] = renamed

    result = collect_native_raw_logits(**arguments)

    assert result.progress["artifact_files"] == [
        {
            "filename": filename,
            "size_bytes": renamed.stat().st_size,
            "sha256": sha256_file(renamed),
        }
    ]


def test_native_raw_logits_rejects_incomplete_or_nonfirst_split(
    tmp_path: Path,
) -> None:
    arguments = _fixture(tmp_path)
    Path(arguments["model_path"]).unlink()
    first = tmp_path / "model-00001-of-00002.gguf"
    first.write_bytes(b"split-model-first")
    arguments["model_path"] = first
    with pytest.raises(ValueError, match="missing split GGUF part"):
        collect_native_raw_logits(**arguments)

    second = tmp_path / "model-00002-of-00002.gguf"
    second.write_bytes(b"split-model-second")
    arguments["model_path"] = second
    with pytest.raises(ValueError, match="must name part 00001"):
        collect_native_raw_logits(**arguments)
