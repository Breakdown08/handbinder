import hashlib
from pathlib import Path

import pytest

from handbinder.models import PALM, Weights, WeightsError, ensure


def test_cached_weights_are_used_without_a_download(tmp_path: Path) -> None:
    (tmp_path / "w.onnx").write_bytes(b"model")
    weights: Weights = Weights("x/w.onnx", hashlib.sha256(b"model").hexdigest())
    assert ensure(weights, tmp_path) == tmp_path / "w.onnx"


def test_corrupted_cache_is_an_error(tmp_path: Path) -> None:
    (tmp_path / PALM.filename).write_bytes(b"truncated")
    with pytest.raises(WeightsError, match="SHA-256"):
        ensure(PALM, tmp_path)
