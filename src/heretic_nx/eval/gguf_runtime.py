"""Small HTTP clients for deterministic GGUF runtime evaluation."""

from __future__ import annotations

from http.client import HTTPConnection, HTTPException, HTTPSConnection
import json
from pathlib import Path
from threading import local
import time
from typing import Any, Mapping
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from heretic_nx.hashing import canonical_json
from heretic_nx.hashing import sha256_file


RESTRICTED_GRAMMAR = "root ::= [ABCD]"


class _PersistentJSONClient:
    """Thread-local HTTP/1.1 connections for a single inference endpoint."""

    def __init__(self, endpoint: str) -> None:
        parsed = urlsplit(endpoint.rstrip("/"))
        if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
            raise ValueError("endpoint must be an absolute HTTP(S) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("endpoint credentials are not supported")
        if parsed.query or parsed.fragment:
            raise ValueError("endpoint must not contain a query or fragment")
        self.endpoint = endpoint.rstrip("/")
        self._scheme = parsed.scheme
        self._host = parsed.hostname
        self._port = parsed.port
        self._base_path = parsed.path.rstrip("/")
        self._thread_state = local()

    def _new_connection(self, timeout: float) -> HTTPConnection:
        connection_type = (
            HTTPSConnection if self._scheme == "https" else HTTPConnection
        )
        return connection_type(self._host, self._port, timeout=timeout)

    def _connection(self, timeout: float) -> HTTPConnection:
        connection = getattr(self._thread_state, "connection", None)
        if connection is None:
            connection = self._new_connection(timeout)
            self._thread_state.connection = connection
        else:
            connection.timeout = timeout
            if connection.sock is not None:
                connection.sock.settimeout(timeout)
        return connection

    def close(self) -> None:
        """Close the connection owned by the calling thread, if any."""

        connection = getattr(self._thread_state, "connection", None)
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass
            finally:
                del self._thread_state.connection

    def request_json(
        self,
        path: str,
        *,
        payload: Mapping[str, Any],
        timeout: float,
        failure: str,
    ) -> Any:
        if not path.startswith("/"):
            raise ValueError("request path must start with a slash")
        body = canonical_json(payload)
        request_path = self._base_path + path
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                connection = self._connection(timeout)
                connection.request(
                    "POST",
                    request_path,
                    body=body,
                    headers={
                        "Accept": "application/json",
                        "Connection": "keep-alive",
                        "Content-Type": "application/json",
                    },
                )
                response = connection.getresponse()
                response_body = response.read()
                if not 200 <= response.status < 300:
                    self.close()
                    status_error = RuntimeError(
                        f"{failure}: HTTP {response.status} {response.reason}"
                    )
                    if response.status in {408, 429} or response.status >= 500:
                        last_error = status_error
                        if attempt < 2:
                            time.sleep(attempt + 1)
                            continue
                        raise RuntimeError(
                            f"{failure} after three attempts"
                        ) from status_error
                    raise status_error
                try:
                    return json.loads(response_body)
                except (json.JSONDecodeError, UnicodeDecodeError) as error:
                    self.close()
                    raise RuntimeError(f"{failure}: invalid JSON response") from error
            except (HTTPException, OSError, TimeoutError) as error:
                last_error = error
                self.close()
                if attempt < 2:
                    time.sleep(attempt + 1)
        raise RuntimeError(f"{failure} after three attempts") from last_error


class NativeRuntimeClient:
    """Deterministic llama.cpp client with one persistent connection per thread."""

    def __init__(self, endpoint: str) -> None:
        self._http = _PersistentJSONClient(endpoint)

    @property
    def endpoint(self) -> str:
        return self._http.endpoint

    def close(self) -> None:
        self._http.close()

    def _content(self, result: Any, *, failure: str) -> str:
        if not isinstance(result, dict) or not isinstance(result.get("content"), str):
            self.close()
            raise RuntimeError(f"{failure}: response omitted string content")
        return result["content"]

    def restricted_choice(self, prompt_tokens: list[int]) -> str:
        payload = {
            "prompt": prompt_tokens,
            "n_predict": 1,
            "temperature": -1,
            "grammar": RESTRICTED_GRAMMAR,
            "stream": False,
        }
        result = self._http.request_json(
            "/completion",
            payload=payload,
            timeout=300,
            failure="llama.cpp native completion failed",
        )
        return self._content(
            result,
            failure="llama.cpp native completion failed",
        )

    def completion(self, prompt_tokens: list[int], *, max_tokens: int) -> str:
        payload = {
            "prompt": prompt_tokens,
            "n_predict": max_tokens,
            "temperature": -1,
            "stream": False,
        }
        result = self._http.request_json(
            "/completion",
            payload=payload,
            timeout=300,
            failure="llama.cpp native generation failed",
        )
        return self._content(
            result,
            failure="llama.cpp native generation failed",
        )


def native_server_properties(endpoint: str) -> dict[str, Any]:
    """Read and validate the identity-bearing llama.cpp ``/props`` fields."""

    request = Request(endpoint.rstrip("/") + "/props", method="GET")
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=30) as response:
                value = json.loads(response.read())
            break
        except (TimeoutError, URLError) as error:
            last_error = error
            if attempt < 2:
                time.sleep(attempt + 1)
    else:
        raise RuntimeError("llama.cpp /props failed after three attempts") from last_error
    if not isinstance(value, dict):
        raise RuntimeError("llama.cpp /props returned a non-object response")
    required = ("model_alias", "model_path", "model_ftype", "build_info")
    missing = [key for key in required if key not in value]
    if missing:
        raise RuntimeError(f"llama.cpp /props omitted identity fields: {missing}")
    if not isinstance(value["model_alias"], str) or not value["model_alias"].strip():
        raise RuntimeError("llama.cpp /props returned an invalid model_alias")
    if not isinstance(value["model_path"], str) or not value["model_path"].strip():
        raise RuntimeError("llama.cpp /props returned an invalid model_path")
    return {key: value[key] for key in required}


def attest_native_model(
    endpoint: str,
    artifact_path: str | Path,
    *,
    expected_model: str | None = None,
) -> dict[str, Any]:
    """Bind one local llama.cpp endpoint to exact, hashed GGUF bytes."""

    artifact = Path(artifact_path).expanduser().resolve(strict=True)
    if not artifact.is_file():
        raise ValueError(f"GGUF artifact is not a file: {artifact}")
    properties = native_server_properties(endpoint)
    served_path = Path(properties["model_path"]).expanduser()
    try:
        served_path = served_path.resolve(strict=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            "llama.cpp /props model_path is not locally resolvable; start the "
            "server with an absolute -m path"
        ) from error
    if not served_path.samefile(artifact):
        raise RuntimeError(
            f"llama.cpp serves {served_path}, not the requested artifact {artifact}"
        )
    if expected_model is not None and properties["model_alias"] != expected_model:
        raise RuntimeError(
            f"llama.cpp model alias {properties['model_alias']!r} does not match "
            f"{expected_model!r}"
        )
    return {
        "endpoint": endpoint.rstrip("/"),
        "model_alias": properties["model_alias"],
        "model_ftype": properties["model_ftype"],
        "model_path": str(artifact),
        "artifact_sha256": sha256_file(artifact),
        "artifact_size_bytes": artifact.stat().st_size,
        "build_info": properties["build_info"],
    }


def require_native_model_identity(
    endpoint: str,
    expected: Mapping[str, Any],
    *,
    verify_artifact_hash: bool = False,
) -> None:
    """Fail if a running endpoint no longer serves its attested model."""

    properties = native_server_properties(endpoint)
    for key in ("model_alias", "model_ftype", "build_info"):
        if properties[key] != expected.get(key):
            raise RuntimeError(
                f"llama.cpp runtime identity changed for {key}: "
                f"{properties[key]!r} != {expected.get(key)!r}"
            )
    served_path = Path(properties["model_path"]).expanduser()
    try:
        served_path = served_path.resolve(strict=True)
        expected_path = Path(str(expected["model_path"])).resolve(strict=True)
    except (FileNotFoundError, KeyError) as error:
        raise RuntimeError("llama.cpp runtime model path is no longer resolvable") from error
    if not served_path.samefile(expected_path):
        raise RuntimeError(
            f"llama.cpp runtime model changed: {served_path} != {expected_path}"
        )
    if verify_artifact_hash:
        actual_size = expected_path.stat().st_size
        actual_sha256 = sha256_file(expected_path)
        if (
            actual_size != expected.get("artifact_size_bytes")
            or actual_sha256 != expected.get("artifact_sha256")
        ):
            raise RuntimeError("attested GGUF artifact bytes changed during evaluation")


def lm_studio_completion(
    endpoint: str,
    model: str,
    prompt: str,
    *,
    max_tokens: int,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    request = Request(
        endpoint.rstrip("/") + "/v1/completions",
        data=canonical_json(payload),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=300) as response:
                result = json.loads(response.read())
            return str(result["choices"][0]["text"])
        except (TimeoutError, URLError) as error:
            last_error = error
            if attempt < 2:
                time.sleep(attempt + 1)
    raise RuntimeError("LM Studio completion failed after three attempts") from last_error


def native_restricted_choice(endpoint: str, prompt_tokens: list[int]) -> str:
    """Return the raw-logit argmax after masking generation to A/B/C/D."""
    payload = {
        "prompt": prompt_tokens,
        "n_predict": 1,
        "temperature": -1,
        "grammar": RESTRICTED_GRAMMAR,
        "stream": False,
    }
    request = Request(
        endpoint.rstrip("/") + "/completion",
        data=canonical_json(payload),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=300) as response:
                result = json.loads(response.read())
            return str(result["content"])
        except (TimeoutError, URLError) as error:
            last_error = error
            if attempt < 2:
                time.sleep(attempt + 1)
    raise RuntimeError("llama.cpp native completion failed after three attempts") from last_error


def native_completion(
    endpoint: str,
    prompt_tokens: list[int],
    *,
    max_tokens: int,
) -> str:
    """Generate greedily from an exact, pre-tokenized llama.cpp prompt."""
    payload = {
        "prompt": prompt_tokens,
        "n_predict": max_tokens,
        "temperature": -1,
        "stream": False,
    }
    request = Request(
        endpoint.rstrip("/") + "/completion",
        data=canonical_json(payload),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=300) as response:
                result = json.loads(response.read())
            return str(result["content"])
        except (TimeoutError, URLError) as error:
            last_error = error
            if attempt < 2:
                time.sleep(attempt + 1)
    raise RuntimeError("llama.cpp native generation failed after three attempts") from last_error
