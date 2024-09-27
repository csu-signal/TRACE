"""
Type stubs for Azure Kinect extension
"""

import numpy as np

class Device:
    def close(self) -> None:
        """Close the device"""
        ...
    def get_frame(self) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
        """
        On success, returns (color image, depth image, dict of body
        tracking info). On fail, returns (None, None, {}). The color
        image is an array of shape (height, width, 4) in BGRA format.
        The depth image is an array of shape (height, width). The body
        tracking dict is in the following format:

        {
            "bodies": [{
                'body_id': unique identifier,
                'joint_positions': [xyz positions],
                'joint_orientation': [wxyz quaternions]
            }, ...],

            "timestamp_usec": timestamp in microseconds
        }
        """
        ...
    def get_calibration_matrices(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (camera_matrix, rotation, translation, distortion).

        camera_matrix --
        """
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
        a fast-forward and will update the frame count.

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
