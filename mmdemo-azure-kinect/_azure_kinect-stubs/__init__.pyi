import numpy as np

class Device:
    def close(self) -> None:
        """Close the device"""
        ...
    def get_frame(self) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
        """
        On success, returns (color image, depth image, dict of body
        tracking info). On fail, returns (None, None, {})
        """
        ...
    def get_calibration_matrices(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns (cameraMatrix, rotation, translation, distortion)"""
        ...
    def get_frame_count(self) -> int:
        """Returns the current frame count of the device"""
        ...

class Playback(Device):
    def __init__(self, path: str):
        """
        Arguments:
        path -- the path to the MKV file of the Azure Kinect Recording
        """
        ...
    def skip_frames(self, n_frames: int) -> None:
        """
        Skip frames in the recording. This works like
        a fast-forward.

        Arguments:
        n_frames -- the number of frames to skip
        """
        ...

class Camera(Device):
    def __init__(self, index: int):
        """
        Arguments:
        index -- the index of the Azure Kinect Camera (usually 0)
        """
        ...
