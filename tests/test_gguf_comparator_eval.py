from __future__ import annotations

import json

from heretic_nx.eval import gguf_runtime


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

    monkeypatch.setattr(gguf_runtime, "urlopen", fake_urlopen)
    choice = gguf_runtime.native_restricted_choice(
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


def test_lm_studio_completion_uses_deterministic_generation(monkeypatch) -> None:
    captured = {}

    class CompletionResponse(_Response):
        def read(self) -> bytes:
            return b'{"choices":[{"text":"answer"}]}'

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return CompletionResponse()

    monkeypatch.setattr(gguf_runtime, "urlopen", fake_urlopen)
    result = gguf_runtime.lm_studio_completion(
        "http://127.0.0.1:1234",
        "abiray-heretic",
        "rendered prompt",
        max_tokens=96,
    )

    payload = json.loads(captured["request"].data)
    assert result == "answer"
    assert captured["request"].full_url.endswith("/v1/completions")
    assert payload == {
        "max_tokens": 96,
        "model": "abiray-heretic",
        "prompt": "rendered prompt",
        "stream": False,
        "temperature": 0,
    }


def test_native_completion_uses_exact_tokens_and_greedy_decoding(monkeypatch) -> None:
    captured = {}

    class NativeResponse(_Response):
        def read(self) -> bytes:
            return b'{"content":"generated"}'

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return NativeResponse()

    monkeypatch.setattr(gguf_runtime, "urlopen", fake_urlopen)
    result = gguf_runtime.native_completion(
        "http://127.0.0.1:1235",
        [124894, 124899],
        max_tokens=96,
    )

    payload = json.loads(captured["request"].data)
    assert result == "generated"
    assert captured["request"].full_url.endswith("/completion")
    assert payload == {
        "n_predict": 96,
        "prompt": [124894, 124899],
        "stream": False,
        "temperature": -1,
    }
