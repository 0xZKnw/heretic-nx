from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from experiments import lfm25_2p6b_gguf_comparator_eval as comparator


class _Response:
    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return b'{"content":"C"}'


def test_native_restricted_choice_uses_token_ids_and_grammar(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(comparator, "urlopen", fake_urlopen)
    choice = comparator.native_restricted_choice(
        "http://127.0.0.1:1235",
        [124894, 124899, 41],
    )

    payload = json.loads(captured["request"].data)
    assert choice == "C"
    assert captured["request"].full_url.endswith("/completion")
    assert captured["timeout"] == 300
    assert payload == {
        "grammar": "root ::= [ABCD]",
        "n_predict": 1,
        "prompt": [124894, 124899, 41],
        "stream": False,
        "temperature": -1,
    }


def test_common_evidence_only_labels_lm_studio_generation_route_when_used() -> None:
    args = SimpleNamespace(
        artifact=Path("comparator.gguf"),
        model="abiray-heretic",
        endpoint="http://127.0.0.1:1234",
    )

    native = comparator.common_evidence(
        args,
        "abc123",
        include_lm_studio=False,
    )
    generation = comparator.common_evidence(
        args,
        "abc123",
        include_lm_studio=True,
    )

    assert "endpoint" not in native
    assert "lm_studio_model_identifier" not in native
    assert generation["endpoint"] == "http://127.0.0.1:1234"
    assert generation["lm_studio_model_identifier"] == "abiray-heretic"
