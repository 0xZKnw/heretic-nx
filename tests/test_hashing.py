from __future__ import annotations

import math

import pytest

from heretic_nx.hashing import canonical_json


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_canonical_json_rejects_non_finite_floats(value: float) -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_json({"nested": [0.0, value]})


def test_canonical_json_remains_deterministic_for_finite_values() -> None:
    assert canonical_json({"z": 1.5, "a": "café"}) == b'{"a":"caf\xc3\xa9","z":1.5}'
