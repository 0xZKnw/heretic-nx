"""Small HTTP clients for deterministic GGUF runtime evaluation."""

from __future__ import annotations

import json
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

from heretic_nx.hashing import canonical_json


RESTRICTED_GRAMMAR = "root ::= [ABCD]"


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
