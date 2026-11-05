import numpy as np
import numpy.typing as npt
from cv2.typing import MatLike

from handbinder.landmarks import crop_square

type Floats = npt.NDArray[np.floating]
type Ints = npt.NDArray[np.int32]


def test_crop_is_padded_to_a_square_and_keeps_its_origin() -> None:
    image: MatLike = np.zeros((100, 200, 3), np.uint8)
    box: Floats = np.array([[50, 10], [90, 30]], dtype=np.float32)
    cropped: tuple[MatLike, Ints, Ints] | None = crop_square(image, box, (0, 0), 1, False)
    assert cropped is not None
    crop: MatLike
    used: Ints
    origin: Ints
    crop, used, origin = cropped
    assert crop.shape == (40, 40, 3)
    assert used.tolist() == [[50, 10], [90, 30]]
    assert origin.tolist() == [50, 0]


def test_crop_outside_the_image_is_none() -> None:
    image: MatLike = np.zeros((100, 200, 3), np.uint8)
    box: Floats = np.array([[300, 300], [320, 320]], dtype=np.float32)
    assert crop_square(image, box, (0, 0), 1, False) is None
