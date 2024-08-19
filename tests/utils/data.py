from pathlib import Path
import pickle

def read_frame_pkl(file: Path):
    """
    Helper function to read frame data

    Returns ColorImageInterface, DepthImageInterface,
    BodyTrackingInterface, CameraCalibrationInterface
    """
    with open(file, "rb") as f:
        color, depth, bt, calibration = pickle.load(f)
        return color, depth, bt, calibration

