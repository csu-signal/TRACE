import pickle
from pathlib import Path


def read_frame_pkl(file: Path):
    """
    Helper function to read frame data

    Returns ColorImageInterface, DepthImageInterface,
    BodyTrackingInterface, CameraCalibrationInterface.

    Note that the CameraCalibrationInterface will have
    `calibration.is_new() == False`
    """
    with open(file, "rb") as f:
        color, depth, bt, calibration = pickle.load(f)

        # calibration will often not be new so tests
        # should still pass with this
        calibration._new = False

        return color, depth, bt, calibration


def read_point_cloud_pkl(file: Path):
    """
    Helper function to read point cloud data.

    Obtained with k4a_transformation_depth_image_to_point_cloud()

    Returns 3d pos of each pixel, DepthImageInterface,
    CameraCalibrationInterface

    Note that the CameraCalibrationInterface will have
    `calibration.is_new() == False`
    """
    with open(file, "rb") as f:
        cloud, depth, calibration = pickle.load(f)

        # calibration will often not be new so tests
        # should still pass with this
        calibration._new = False

        return cloud, depth, calibration
