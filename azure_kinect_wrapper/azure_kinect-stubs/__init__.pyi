
from abc import ABC, abstractmethod
import numpy as np
from typing import final


class Device(ABC):
    @abstractmethod
    def close(self) -> None:
        ...

    @abstractmethod
    def get_frame(self) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
        ...

    @abstractmethod
    def get_calibration_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ...


@final
class Playback(Device):
    def __init__(self, path: str):
        ...

    def close(self) -> None:
        ...

    def get_frame(self) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
        ...

    def get_calibration_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ...


@final
class Camera(Device):
    def __init__(self, index: int):
        ...

    def close(self) -> None:
        ...

    def get_frame(self) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
        ...

    def get_calibration_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ...
