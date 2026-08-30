from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Barrier

import pytest

from heretic_nx.eval import gguf_runtime


class _Response:
    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return b'{"content":"C"}'


class _PersistentResponse:
    status = 200
    reason = "OK"

    def __init__(self, content: str) -> None:
        self._content = content

    def read(self) -> bytes:
        return json.dumps({"content": self._content}).encode()


class _RawPersistentResponse:
    def __init__(
        self,
        body: bytes,
        *,
        status: int = 200,
        reason: str = "OK",
    ) -> None:
        self._body = body
        self.status = status
        self.reason = reason

    def read(self) -> bytes:
        return self._body


class _Socket:
    def __init__(self) -> None:
        self.timeouts = []

    def settimeout(self, timeout: float) -> None:
        self.timeouts.append(timeout)


class _PersistentConnection:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.requests = []
        self.timeout = None
        self.sock = _Socket()
        self.closed = False

    def request(self, method, path, *, body, headers) -> None:
        self.requests.append((method, path, body, headers))

    def getresponse(self) -> _PersistentResponse:
        return _PersistentResponse(self.responses.pop(0))

    def close(self) -> None:
        self.closed = True
        self.sock = None


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


def test_native_runtime_client_reuses_one_connection_per_thread(monkeypatch) -> None:
    connections = []

    def connection_factory(host, port, timeout):
        assert host == "127.0.0.1"
        assert port == 1235
        connection = _PersistentConnection(["C", "A"])
        connection.timeout = timeout
        connections.append(connection)
        return connection

    monkeypatch.setattr(gguf_runtime, "HTTPConnection", connection_factory)
    client = gguf_runtime.NativeRuntimeClient("http://127.0.0.1:1235")

    assert client.restricted_choice([124894, 10]) == "C"
    assert client.restricted_choice([124894, 20]) == "A"

    assert len(connections) == 1
    assert len(connections[0].requests) == 2
    first = connections[0].requests[0]
    assert first[0:2] == ("POST", "/completion")
    assert json.loads(first[2]) == {
        "grammar": "root ::= [ABCD]",
        "n_predict": 1,
        "prompt": [124894, 10],
        "stream": False,
        "temperature": -1,
    }
    assert first[3]["Connection"] == "keep-alive"
    client.close()
    assert connections[0].closed is True


def test_native_runtime_client_never_shares_connection_between_threads(
    monkeypatch,
) -> None:
    connections = []

    def connection_factory(_host, _port, timeout):
        connection = _PersistentConnection(["B"])
        connection.timeout = timeout
        connections.append(connection)
        return connection

    monkeypatch.setattr(gguf_runtime, "HTTPConnection", connection_factory)
    client = gguf_runtime.NativeRuntimeClient("http://127.0.0.1:1235")
    barrier = Barrier(2)

    def worker(token: int) -> str:
        barrier.wait()
        return client.restricted_choice([token])

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(worker, [1, 2]))

    assert results == ["B", "B"]
    assert len(connections) == 2
    assert all(len(connection.requests) == 1 for connection in connections)


def test_native_runtime_client_reconnects_after_transport_failure(
    monkeypatch,
) -> None:
    connections = []

    class FailingConnection(_PersistentConnection):
        def request(self, method, path, *, body, headers) -> None:
            super().request(method, path, body=body, headers=headers)
            raise ConnectionResetError("connection reset")

    def connection_factory(_host, _port, timeout):
        if not connections:
            connection = FailingConnection([])
        else:
            connection = _PersistentConnection(["D"])
        connection.timeout = timeout
        connections.append(connection)
        return connection

    monkeypatch.setattr(gguf_runtime, "HTTPConnection", connection_factory)
    monkeypatch.setattr(gguf_runtime.time, "sleep", lambda _seconds: None)
    client = gguf_runtime.NativeRuntimeClient("http://127.0.0.1:1235")

    assert client.restricted_choice([10, 20]) == "D"
    assert len(connections) == 2
    assert connections[0].closed is True


def test_native_runtime_client_supports_https_and_endpoint_base_path(
    monkeypatch,
) -> None:
    connections = []

    def connection_factory(host, port, timeout):
        assert host == "example.test"
        assert port == 8443
        connection = _PersistentConnection(["generated"])
        connection.timeout = timeout
        connections.append(connection)
        return connection

    monkeypatch.setattr(gguf_runtime, "HTTPSConnection", connection_factory)
    client = gguf_runtime.NativeRuntimeClient(
        "https://example.test:8443/llama/api/"
    )

    assert client.completion([7, 8], max_tokens=17) == "generated"
    assert len(connections) == 1
    request = connections[0].requests[0]
    assert request[0:2] == ("POST", "/llama/api/completion")
    assert json.loads(request[2]) == {
        "n_predict": 17,
        "prompt": [7, 8],
        "stream": False,
        "temperature": -1,
    }


def test_native_runtime_client_closes_connection_on_http_error(monkeypatch) -> None:
    connection = _PersistentConnection([])
    connection.getresponse = lambda: _RawPersistentResponse(
        b'{"error":"bad request"}',
        status=400,
        reason="Bad Request",
    )
    monkeypatch.setattr(
        gguf_runtime,
        "HTTPConnection",
        lambda _host, _port, timeout: connection,
    )
    client = gguf_runtime.NativeRuntimeClient("http://127.0.0.1:1235")

    with pytest.raises(RuntimeError, match="HTTP 400 Bad Request"):
        client.restricted_choice([10])
    assert connection.closed is True


def test_native_runtime_client_retries_transient_http_error(monkeypatch) -> None:
    connections = []

    def connection_factory(_host, _port, timeout):
        connection = _PersistentConnection([])
        connection.timeout = timeout
        connection.getresponse = lambda: _RawPersistentResponse(
            b'{"error":"busy"}',
            status=503,
            reason="Service Unavailable",
        )
        connections.append(connection)
        return connection

    monkeypatch.setattr(gguf_runtime, "HTTPConnection", connection_factory)
    monkeypatch.setattr(gguf_runtime.time, "sleep", lambda _seconds: None)
    client = gguf_runtime.NativeRuntimeClient("http://127.0.0.1:1235")

    with pytest.raises(RuntimeError, match="after three attempts"):
        client.restricted_choice([10])
    assert len(connections) == 3
    assert all(connection.closed for connection in connections)


@pytest.mark.parametrize(
    "body, message",
    [
        (b"not-json", "invalid JSON response"),
        (b'{"content":42}', "response omitted string content"),
    ],
)
def test_native_runtime_client_rejects_malformed_json_response(
    monkeypatch,
    body: bytes,
    message: str,
) -> None:
    connection = _PersistentConnection([])
    connection.getresponse = lambda: _RawPersistentResponse(body)
    monkeypatch.setattr(
        gguf_runtime,
        "HTTPConnection",
        lambda _host, _port, timeout: connection,
    )
    client = gguf_runtime.NativeRuntimeClient("http://127.0.0.1:1235")

    with pytest.raises(RuntimeError, match=message):
        client.restricted_choice([10])
    assert connection.closed is True


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


def test_native_model_attestation_binds_props_to_exact_file(
    tmp_path: Path, monkeypatch
) -> None:
    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(b"exact-model-bytes")

    class PropsResponse(_Response):
        def read(self) -> bytes:
            return json.dumps(
                {
                    "model_alias": "release-q4",
                    "model_path": str(artifact),
                    "model_ftype": "Q4_K_M",
                    "build_info": "b10621-deadbeef",
                }
            ).encode()

    monkeypatch.setattr(gguf_runtime, "urlopen", lambda _request, timeout: PropsResponse())

    identity = gguf_runtime.attest_native_model(
        "http://127.0.0.1:1235",
        artifact,
        expected_model="release-q4",
    )
    gguf_runtime.require_native_model_identity(
        "http://127.0.0.1:1235",
        identity,
    )

    assert identity["artifact_size_bytes"] == len(b"exact-model-bytes")
    assert len(identity["artifact_sha256"]) == 64
    assert identity["model_ftype"] == "Q4_K_M"

    artifact.write_bytes(b"mutated-model-bytes")
    with pytest.raises(RuntimeError, match="changed during evaluation"):
        gguf_runtime.require_native_model_identity(
            "http://127.0.0.1:1235",
            identity,
            verify_artifact_hash=True,
        )


def test_native_model_attestation_rejects_wrong_served_file(
    tmp_path: Path, monkeypatch
) -> None:
    expected = tmp_path / "expected.gguf"
    served = tmp_path / "served.gguf"
    expected.write_bytes(b"expected")
    served.write_bytes(b"served")

    class PropsResponse(_Response):
        def read(self) -> bytes:
            return json.dumps(
                {
                    "model_alias": "model",
                    "model_path": str(served),
                    "model_ftype": "Q8_0",
                    "build_info": "test",
                }
            ).encode()

    monkeypatch.setattr(gguf_runtime, "urlopen", lambda _request, timeout: PropsResponse())

    with pytest.raises(RuntimeError, match="not the requested artifact"):
        gguf_runtime.attest_native_model("http://127.0.0.1:1235", expected)
