"""Debug view: what the pipeline sees. Nothing is drawn or kept unless --preview is on."""

import cv2 as cv
import numpy as np
import numpy.typing as npt
from cv2.typing import MatLike

from handbinder.gestures import Gesture, Pose
from handbinder.landmarks import Hand

QUIT: int = ord("q")
WHITE: tuple[int, int, int] = (255, 255, 255)
GREEN: tuple[int, int, int] = (0, 200, 0)
FINGER_STARTS: tuple[int, ...] = (1, 5, 9, 13, 17)
BONES: tuple[tuple[int, int], ...] = tuple(
    pair for s in FINGER_STARTS for pair in ((0, s), (s, s + 1), (s + 1, s + 2), (s + 2, s + 3))
)


def draw(frame: MatLike, hand: Hand | None, pose: Pose | None, gesture: Gesture | None) -> MatLike:
    out: MatLike = frame.copy()
    if hand is not None:
        points: npt.NDArray[np.int32] = hand.points[:, :2].astype(np.int32)
        a: int
        b: int
        for a, b in BONES:
            cv.line(out, tuple(points[a].tolist()), tuple(points[b].tolist()), WHITE, 2)
        point: npt.NDArray[np.int32]
        for point in points:
            cv.circle(out, tuple(point.tolist()), 3, GREEN, -1)
    if pose is not None:
        text: str = f"thumb {pose.thumb_angle:.0f} deg, curled {pose.curled}/4"
        cv.putText(out, text, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, GREEN)
    if gesture is not None:
        cv.putText(out, gesture, (10, 55), cv.FONT_HERSHEY_DUPLEX, 0.8, GREEN)
    return out
