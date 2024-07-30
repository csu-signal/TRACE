import os
from config import K4A_DIR

# tell the script where to find certain dll's for k4a, cuda, etc.
# body tracking sdk's tools should contain everything
os.add_dll_directory(K4A_DIR)
import azure_kinect

from typing import final
import numpy as np

SHAPE = (1200, 800)


@final
class FakeCamera:
    @property
    def __class__(self): #pyright: ignore
        return azure_kinect.Device

    def get_frame(self) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
        return (np.zeros((*SHAPE, 4), dtype=np.float32), np.zeros(SHAPE, dtype=np.float32), {"bodies": []})

    def get_calibration_matrices(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return np.zeros((3, 3)), np.zeros((3, 3)), np.zeros(3), np.zeros(8)

    def close(self) -> None:
        return
