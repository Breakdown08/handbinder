"""Gestures as geometry of the 21 world points."""

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt

type Floats = npt.NDArray[np.floating]

WRIST: int = 0
THUMB_BASE: int = 2
THUMB_TIP: int = 4
# (middle joint, tip) of the index, middle, ring and little fingers.
FINGERS: tuple[tuple[int, int], ...] = ((6, 8), (10, 12), (14, 16), (18, 20))
UP: Floats = np.array([0, -1, 0], dtype=np.float32)
THUMB_UP_MAX_ANGLE: float = 40


class Gesture(StrEnum):
    THUMB_UP = "thumb_up"


@dataclass(frozen=True)
class Pose:
    thumb_angle: float  # degrees between the thumb and straight up
    curled: int  # fingers bent into the palm, thumb excluded


def measure(world: Floats) -> Pose:
    thumb: Floats = world[THUMB_TIP] - world[THUMB_BASE]
    cosine: float = float(thumb @ UP) / max(float(np.linalg.norm(thumb)), 1e-6)
    wrist: Floats = world[WRIST]
    curled: int = sum(
        float(np.linalg.norm(world[tip] - wrist)) < float(np.linalg.norm(world[joint] - wrist))
        for joint, tip in FINGERS
    )
    return Pose(math.degrees(math.acos(min(max(cosine, -1), 1))), curled)


def classify(pose: Pose) -> Gesture | None:
    if pose.curled == len(FINGERS) and pose.thumb_angle <= THUMB_UP_MAX_ANGLE:
        return Gesture.THUMB_UP
    return None
