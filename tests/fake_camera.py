from typing import final
import numpy as np

# (rows, cols)
SHAPE = (1080, 1920)

@final
class FakeCamera:
    """
    Adheres to the azure kinect Device interface but does not
    return any data. To be used for testing purposes.
    """
    def __init__(self) -> None:
        self.frame = 0

    def get_frame(self) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
        return (np.zeros((*SHAPE, 4), dtype=np.uint8), np.zeros(SHAPE, dtype=np.float32), {"bodies": []})

    def get_calibration_matrices(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return np.zeros((3, 3)), np.zeros((3, 3)), np.zeros(3), np.zeros(8)

    def close(self) -> None:
        return

    def get_frame_count(self) -> int:
        self.frame += 1
        return self.frame
