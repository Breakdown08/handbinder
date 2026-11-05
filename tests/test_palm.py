import numpy as np
import numpy.typing as npt

from handbinder.palm import Palm, decode, make_anchors

type Floats = npt.NDArray[np.floating]


def test_anchors_cover_both_feature_maps() -> None:
    anchors: Floats = make_anchors()
    assert anchors.shape == (2016, 2)
    assert anchors[0].tolist() == anchors[1].tolist()
    np.testing.assert_allclose(anchors[0], [1 / 48, 1 / 48])
    np.testing.assert_allclose(anchors[2], [3 / 48, 1 / 48])
    np.testing.assert_allclose(anchors[1152], [1 / 24, 1 / 24])
    np.testing.assert_allclose(anchors[-1], [23 / 24, 23 / 24])


def test_decode_takes_the_best_score_into_frame_pixels() -> None:
    anchors: Floats = np.array([[0.5, 0.5], [0.25, 0.75]], dtype=np.float32)
    deltas: Floats = np.zeros((2, 18), dtype=np.float32)
    deltas[1, 2:4] = 96  # half of the 192 input
    scores: Floats = np.array([-5, 5], dtype=np.float32)
    palm: Palm | None = decode(deltas, scores, anchors, 640, np.array([0, 80], dtype=np.float32))
    assert palm is not None
    np.testing.assert_allclose(palm.box, [[0, 240], [320, 560]])
    np.testing.assert_allclose(palm.points[0], [160, 400])


def test_decode_rejects_a_low_score() -> None:
    anchors: Floats = np.array([[0.5, 0.5]], dtype=np.float32)
    deltas: Floats = np.zeros((1, 18), dtype=np.float32)
    assert decode(deltas, np.array([-1], dtype=np.float32), anchors, 640, np.zeros(2, np.float32)) is None
