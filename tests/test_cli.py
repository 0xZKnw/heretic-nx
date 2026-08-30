from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

from heretic_nx.cli import doctor, main


gguf = pytest.importorskip("gguf")
from gguf import GGMLQuantizationType, GGUFWriter  # noqa: E402
from gguf.quants import quantize  # noqa: E402


def _write_q8(path: Path) -> None:
    writer = GGUFWriter(path, "llama")
    writer.add_name("heretic-nx-cli-test")
    values = np.linspace(-1, 1, 128, dtype=np.float32).reshape(4, 32)
    writer.add_tensor(
        "blk.0.attn_output.weight",
        quantize(values, GGMLQuantizationType.Q8_0),
        raw_dtype=GGMLQuantizationType.Q8_0,
    )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_doctor_reports_gguf_and_k_codec_status() -> None:
    report = doctor("auto")

    assert report["libraries"]["gguf"] != "not-installed"
    assert isinstance(report["k_quant_codec"]["available"], bool)


def test_inspect_gguf_cli_writes_machine_readable_report(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    source = tmp_path / "model.gguf"
    output = tmp_path / "inspection.json"
    _write_q8(source)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hnx",
            "inspect-gguf",
            "--input",
            str(source),
            "--output",
            str(output),
        ],
    )

    main()

    printed = json.loads(capsys.readouterr().out)
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert printed == persisted
    assert printed["editable_tensor_count"] == 1
    assert printed["tensors"][0]["quantization"] == "Q8_0"
