import numpy as np
import numpy.typing as npt

from handbinder.gestures import Gesture, Pose, classify, measure

type Floats = npt.NDArray[np.floating]


def fist(thumb_tip: tuple[float, float, float]) -> Floats:
    """A fist in metres, y down: four tips closer to the wrist than their middle joints."""
    world: Floats = np.zeros((21, 3), dtype=np.float32)
    world[0] = (0, 0.08, 0)
    world[2] = (0.03, 0, 0)
    world[4] = thumb_tip
    joint: int
    for joint in (6, 10, 14, 18):
        world[joint] = (0, 0, 0)
        world[joint + 2] = (0, 0.04, 0)
    return world


def test_thumb_straight_up_on_a_fist() -> None:
    pose: Pose = measure(fist((0.03, -0.06, 0)))
    assert pose.curled == 4
    assert pose.thumb_angle < 1
    assert classify(pose) is Gesture.THUMB_UP


def test_sideways_thumb_is_not_thumb_up() -> None:
    pose: Pose = measure(fist((0.09, 0, 0)))
    assert 89 < pose.thumb_angle < 91
    assert classify(pose) is None


def test_open_fingers_are_not_thumb_up() -> None:
    assert classify(Pose(thumb_angle=0, curled=3)) is None


def test_y_axis_is_down() -> None:
    assert classify(measure(fist((0.03, 0.06, 0)))) is None
