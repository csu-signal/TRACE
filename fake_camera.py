from typing import final
import numpy as np

SHAPE = (1920, 1080)

@final
class FakeCamera:
    def get_frame(self) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
        return (np.zeros((*SHAPE, 4), dtype=np.float32), np.zeros(SHAPE, dtype=np.float32), {"bodies": []})

    def get_calibration_matrices(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return np.zeros((3, 3)), np.zeros((3, 3)), np.zeros(3), np.zeros(8)

    def close(self) -> None:
        return

    def get_frame_count(self) -> int:
        return 0
