import numpy as np
import numpy.typing as npt
from cv2.typing import MatLike

from handbinder.landmarks import Hand
from handbinder.palm import Palm
from handbinder.tracker import HandTracker, palm_from_hand

type Floats = npt.NDArray[np.floating]

FRAME: MatLike = np.zeros((4, 4, 3), np.uint8)


def hand() -> Hand:
    points: Floats = np.zeros((21, 3), dtype=np.float32)
    points[:, 0] = np.arange(21)
    points[:, 1] = np.arange(21) * 2
    return Hand(points, np.zeros((21, 3), dtype=np.float32), 0.95)


class FakeDetector:
    def __init__(self) -> None:
        self.calls: int = 0

    def detect(self, frame: MatLike) -> Palm | None:
        self.calls += 1
        return palm_from_hand(hand())


class FakeLandmarker:
    def __init__(self, answers: list[Hand | None]) -> None:
        self.answers: list[Hand | None] = answers

    def estimate(self, frame: MatLike, palm: Palm) -> Hand | None:
        return self.answers.pop(0)


def test_palm_points_follow_the_detector_order() -> None:
    palm: Palm = palm_from_hand(hand())
    assert palm.points[:, 0].tolist() == [0, 5, 9, 13, 17, 1, 2]
    assert palm.box.tolist() == [[0, 0], [17, 34]]


def test_detector_runs_only_until_the_hand_is_found_and_after_it_is_lost() -> None:
    detector: FakeDetector = FakeDetector()
    tracker: HandTracker = HandTracker(detector, FakeLandmarker([hand(), hand(), None, hand()]))
    assert not tracker.active
    tracker.update(FRAME)
    tracker.update(FRAME)
    assert detector.calls == 1
    assert tracker.update(FRAME) is None
    assert not tracker.active
    tracker.update(FRAME)
    assert detector.calls == 2
    assert tracker.active
