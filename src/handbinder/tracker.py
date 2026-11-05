"""Finds a hand once, then follows it: the landmarks of one frame seed the next, without the detector."""

from typing import Protocol

import numpy as np
import numpy.typing as npt
from cv2.typing import MatLike

from handbinder.landmarks import Hand
from handbinder.palm import Palm

type Floats = npt.NDArray[np.floating]

# Hand points the palm detector reports, in its order.
PALM_POINTS: tuple[int, ...] = (0, 5, 9, 13, 17, 1, 2)


class Detector(Protocol):
    def detect(self, frame: MatLike) -> Palm | None: ...


class Landmarker(Protocol):
    def estimate(self, frame: MatLike, palm: Palm) -> Hand | None: ...


def palm_from_hand(hand: Hand) -> Palm:
    points: Floats = hand.points[list(PALM_POINTS), :2]
    box: Floats = np.array([points.min(axis=0), points.max(axis=0)], dtype=np.float32)
    return Palm(box, points, hand.score)


class HandTracker:
    def __init__(self, detector: Detector, landmarker: Landmarker) -> None:
        self._detector: Detector = detector
        self._landmarker: Landmarker = landmarker
        self._last: Hand | None = None

    @property
    def active(self) -> bool:
        return self._last is not None

    def update(self, frame: MatLike) -> Hand | None:
        palm: Palm | None = palm_from_hand(self._last) if self._last else self._detector.detect(frame)
        self._last = self._landmarker.estimate(frame, palm) if palm else None
        return self._last
