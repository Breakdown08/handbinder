"""The camera loop: slow while the frame is empty, faster once a hand is in it."""

import logging
import time
from pathlib import Path

import cv2 as cv
from cv2.typing import MatLike

from handbinder import models
from handbinder.config import Config
from handbinder.gestures import Gesture, Pose, classify, measure
from handbinder.landmarks import Hand, HandLandmarker
from handbinder.palm import PalmDetector
from handbinder.preview import QUIT, draw
from handbinder.runner import ScriptRunner
from handbinder.tracker import HandTracker
from handbinder.trigger import Trigger

log: logging.Logger = logging.getLogger(__name__)
WINDOW: str = "handbinder"


class CameraError(RuntimeError):
    pass


def run(config: Config, preview: bool, cache: Path | None = None) -> None:
    cv.setNumThreads(config.threads)
    tracker: HandTracker = HandTracker(
        PalmDetector(models.ensure(models.PALM, cache)),
        HandLandmarker(models.ensure(models.LANDMARKS, cache)),
    )
    trigger: Trigger = Trigger(config.timings())
    runner: ScriptRunner = ScriptRunner()
    camera: cv.VideoCapture = cv.VideoCapture(config.device)
    if not camera.isOpened():
        raise CameraError(f"camera {config.device} cannot be opened")
    log.info("watching camera %d; frames stay in memory", config.device)
    next_at: float = 0
    try:
        while True:
            # grab() without retrieve() keeps the camera buffer fresh and does not decode the frame.
            if not camera.grab():
                raise CameraError(f"camera {config.device} stopped giving frames")
            now: float = time.monotonic()
            if now < next_at:
                continue
            next_at = now + 1 / (config.active_fps if tracker.active else config.idle_fps)
            ok: bool
            frame: MatLike
            ok, frame = camera.retrieve()
            if not ok:
                continue
            hand: Hand | None = tracker.update(frame)
            pose: Pose | None = measure(hand.world) if hand else None
            gesture: Gesture | None = classify(pose) if pose else None
            fired: Gesture | None = trigger.update(gesture, now)
            if fired:
                runner.run(fired, config.gestures[fired].script)
            if preview:
                cv.imshow(WINDOW, draw(frame, hand, pose, gesture))
                if cv.waitKey(1) & 0xFF == QUIT:
                    return
    finally:
        camera.release()
        if preview:
            cv.destroyAllWindows()
