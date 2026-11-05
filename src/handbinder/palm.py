"""Palm detector: an SSD over 2016 anchors, run through cv.dnn.

Adapted from opencv_zoo, see NOTICE. Only the best palm is kept, so NMS is not needed.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np
import numpy.typing as npt
from cv2.typing import MatLike

type Floats = npt.NDArray[np.floating]

INPUT_SIZE: int = 192
SCORE_THRESHOLD: float = 0.6
# Grid side and anchors per cell of the two feature maps.
ANCHOR_LEVELS: tuple[tuple[int, int], ...] = ((24, 2), (12, 6))
POINTS: int = 7


@dataclass(frozen=True)
class Palm:
    box: Floats  # [[x1, y1], [x2, y2]] in frame pixels
    points: Floats  # 7 × 2: wrist, finger bases, thumb base, in frame pixels
    score: float


def make_anchors() -> Floats:
    """Anchor centres in 0..1, row by row, level by level."""
    out: list[tuple[float, float]] = []
    grid: int
    repeat: int
    for grid, repeat in ANCHOR_LEVELS:
        y: int
        x: int
        for y in range(grid):
            for x in range(grid):
                out += [((x + 0.5) / grid, (y + 0.5) / grid)] * repeat
    return np.array(out, dtype=np.float32)


def decode(deltas: Floats, scores: Floats, anchors: Floats, scale: float, pad: Floats) -> Palm | None:
    """Best detection in frame pixels. `scale` is the longest frame side, `pad` the padding in pixels."""
    best: int = int(np.argmax(scores))
    score: float = 1 / (1 + math.exp(-float(scores[best])))
    if score < SCORE_THRESHOLD:
        return None
    delta: Floats = deltas[best] / INPUT_SIZE
    anchor: Floats = anchors[best]
    half: Floats = delta[2:4] / 2
    centre: Floats = delta[:2] + anchor
    box: Floats = np.array([centre - half, centre + half], dtype=np.float32) * scale - pad
    points: Floats = (delta[4:].reshape(POINTS, 2) + anchor) * scale - pad
    return Palm(box, points, score)


class PalmDetector:
    def __init__(self, model: Path) -> None:
        self._net: cv.dnn.Net = cv.dnn.readNet(str(model))
        self._anchors: Floats = make_anchors()

    def detect(self, frame: MatLike) -> Palm | None:
        height: int = frame.shape[0]
        width: int = frame.shape[1]
        longest: int = max(height, width)
        ratio: float = INPUT_SIZE / longest
        new_width: int = int(width * ratio)
        new_height: int = int(height * ratio)
        left: int = (INPUT_SIZE - new_width) // 2
        top: int = (INPUT_SIZE - new_height) // 2
        image: MatLike = cv.resize(frame, (new_width, new_height))
        image = cv.copyMakeBorder(
            image,
            top,
            INPUT_SIZE - new_height - top,
            left,
            INPUT_SIZE - new_width - left,
            cv.BORDER_CONSTANT,
            None,
            (0, 0, 0),
        )
        blob: Floats = cv.cvtColor(image, cv.COLOR_BGR2RGB).astype(np.float32)[np.newaxis] / 255
        self._net.setInput(blob)
        deltas: MatLike
        scores: MatLike
        deltas, scores = self._net.forward(self._net.getUnconnectedOutLayersNames())
        pad: Floats = np.array([left, top], dtype=np.float32) / ratio
        return decode(
            np.asarray(deltas, np.float32)[0],
            np.asarray(scores, np.float32)[0, :, 0],
            self._anchors,
            longest,
            np.trunc(pad),
        )
