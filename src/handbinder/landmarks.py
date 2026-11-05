"""21 hand points: the palm is rotated upright, cropped and passed through cv.dnn.

Adapted from opencv_zoo, see NOTICE.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np
import numpy.typing as npt
from cv2.typing import MatLike

from handbinder.palm import Palm

type Floats = npt.NDArray[np.floating]
type Ints = npt.NDArray[np.int32]

INPUT_SIZE: int = 224
CONFIDENCE_THRESHOLD: float = 0.9
WRIST: int = 0
MIDDLE_BASE: int = 2  # in the 7 palm points
PRE_ENLARGE: float = 4  # generous crop, big enough to survive the rotation
PALM_ENLARGE: float = 3  # palm box to hand box, with the shift towards the fingers
PALM_SHIFT: tuple[float, float] = (0, -0.4)


@dataclass(frozen=True)
class Hand:
    points: Floats  # 21 × 3: frame pixels, z relative to the wrist
    world: Floats  # 21 × 3: metres around the hand centre, y down
    score: float


def crop_square(
    image: MatLike, box: Floats, shift: tuple[float, float], enlarge: float, for_rotation: bool
) -> tuple[MatLike, Ints, Ints] | None:
    """Crop around `box` and pad to a square; None when the box lies outside the image.

    Returns the crop, the box used in `image` pixels and the crop origin in `image` pixels.
    """
    size: Floats = box[1] - box[0]
    moved: Floats = box + np.array(shift, dtype=np.float32) * size
    centre: Floats = moved.sum(axis=0) / 2
    half: Floats = (moved[1] - moved[0]) * enlarge / 2
    clipped: Ints = np.array([centre - half, centre + half]).astype(np.int32)
    clipped[:, 0] = np.clip(clipped[:, 0], 0, image.shape[1])
    clipped[:, 1] = np.clip(clipped[:, 1], 0, image.shape[0])
    crop: MatLike = image[clipped[0, 1] : clipped[1, 1], clipped[0, 0] : clipped[1, 0]]
    if crop.size == 0:
        return None
    height: int = crop.shape[0]
    width: int = crop.shape[1]
    side: int = int(math.hypot(height, width)) if for_rotation else max(height, width)
    left: int = (side - width) // 2
    top: int = (side - height) // 2
    crop = cv.copyMakeBorder(
        crop, top, side - height - top, left, side - width - left, cv.BORDER_CONSTANT, None, (0, 0, 0)
    )
    origin: Ints = clipped[0] - np.array([left, top], dtype=np.int32)
    return crop, clipped, origin


class HandLandmarker:
    def __init__(self, model: Path) -> None:
        self._net: cv.dnn.Net = cv.dnn.readNet(str(model))

    def estimate(self, frame: MatLike, palm: Palm) -> Hand | None:
        cropped: tuple[MatLike, Ints, Ints] | None = crop_square(frame, palm.box, (0, 0), PRE_ENLARGE, True)
        if cropped is None:
            return None
        image: MatLike
        crop_box: Ints
        origin: Ints
        image, crop_box, origin = cropped
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Turn the hand so that the fingers point up, as the model expects.
        points: Floats = palm.points - origin
        direction: Floats = points[MIDDLE_BASE] - points[WRIST]
        radians: float = math.pi / 2 - math.atan2(-direction[1], direction[0])
        radians -= 2 * math.pi * math.floor((radians + math.pi) / (2 * math.pi))
        angle: float = math.degrees(radians)
        corners: Ints = crop_box - origin
        centre: tuple[float, float] = (float(corners[:, 0].sum() / 2), float(corners[:, 1].sum() / 2))
        rotation: MatLike = cv.getRotationMatrix2D(centre, angle, 1.0)
        rotated: MatLike = cv.warpAffine(image, rotation, (image.shape[1], image.shape[0]))
        upright: Floats = cv.transform(points[np.newaxis], rotation)[0]
        upright_box: Floats = np.array([upright.min(axis=0), upright.max(axis=0)], dtype=np.float32)

        hand_crop: tuple[MatLike, Ints, Ints] | None = crop_square(
            rotated, upright_box, PALM_SHIFT, PALM_ENLARGE, False
        )
        if hand_crop is None:
            return None
        crop: MatLike = hand_crop[0]
        hand_box: Ints = hand_crop[1]
        resized: MatLike = cv.resize(crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv.INTER_AREA)
        self._net.setInput(resized.astype(np.float32)[np.newaxis] / 255)
        outputs: list[MatLike] = list(self._net.forward(self._net.getUnconnectedOutLayersNames()))
        confidence: float = float(outputs[1][0][0])
        if confidence < CONFIDENCE_THRESHOLD:
            return None

        # Back to frame pixels: model input → rotated image → padded crop → frame.
        raw: Floats = np.asarray(outputs[0], np.float32)[0].reshape(-1, 3)
        scale: float = float(max(hand_box[1] - hand_box[0])) / INPUT_SIZE
        offset: Floats = hand_box.sum(axis=0).astype(np.float32) / 2
        in_rotated: Floats = (raw[:, :2] - INPUT_SIZE / 2) * scale + offset
        back: MatLike = cv.invertAffineTransform(rotation)
        xy: Floats = cv.transform(in_rotated[np.newaxis], back)[0] + origin
        z: Floats = raw[:, 2] * scale

        # World points keep their scale; only the rotation is undone.
        raw_world: Floats = np.asarray(outputs[3], np.float32)[0].reshape(-1, 3)
        turn: Floats = cv.getRotationMatrix2D((0, 0), angle, 1.0)[:, :2].astype(np.float32)
        world_xy: Floats = raw_world[:, :2] @ turn
        return Hand(np.c_[xy, z], np.c_[world_xy, raw_world[:, 2]], confidence)
