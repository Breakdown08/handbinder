"""ONNX weights from opencv_zoo, pinned by commit and SHA-256, cached outside the repository."""

import hashlib
import logging
import os
import shutil
import urllib.request
from dataclasses import dataclass
from http.client import HTTPResponse
from io import BufferedWriter
from pathlib import Path

log: logging.Logger = logging.getLogger(__name__)

ZOO_COMMIT: str = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
BASE_URL: str = f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{ZOO_COMMIT}/models"


@dataclass(frozen=True)
class Weights:
    path: str
    sha256: str

    @property
    def filename(self) -> str:
        return self.path.rsplit("/", 1)[1]


PALM: Weights = Weights(
    "palm_detection_mediapipe/palm_detection_mediapipe_2023feb.onnx",
    "78ff51c38496b7fc8b8ebdb6cc8c1abb02fa6c38427c6848254cdaba57fcce7c",
)
LANDMARKS: Weights = Weights(
    "handpose_estimation_mediapipe/handpose_estimation_mediapipe_2023feb.onnx",
    "db0898ae717b76b075d9bf563af315b29562e11f8df5027a1ef07b02bef6d81c",
)


class WeightsError(RuntimeError):
    pass


def cache_dir() -> Path:
    base: str | None = os.environ.get("XDG_CACHE_HOME")
    return (Path(base) if base else Path.home() / ".cache") / "handbinder"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure(weights: Weights, directory: Path | None = None) -> Path:
    """Path to verified weights, downloading them on first use."""
    target: Path = (directory or cache_dir()) / weights.filename
    if target.exists():
        if sha256(target) == weights.sha256:
            return target
        raise WeightsError(f"{target} does not match its SHA-256; delete it to download again")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial: Path = target.with_suffix(".part")
    url: str = f"{BASE_URL}/{weights.path}"
    log.info("downloading %s", url)
    response: HTTPResponse
    out: BufferedWriter
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out)
    if sha256(partial) != weights.sha256:
        partial.unlink()
        raise WeightsError(f"{url} does not match its SHA-256")
    partial.rename(target)
    return target
