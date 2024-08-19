import json
import os
from pathlib import Path
from typing import final

import numpy as np
import pytest
from PIL import Image

from mmdemo.features.gesture.gesture_feature import Gesture
from mmdemo.interfaces import (
    BodyTrackingInterface,
    CameraCalibrationInterface,
    ColorImageInterface,
    DepthImageInterface,
    GestureConesInterface,
)
from mmdemo.utils.camera_calibration_utils import (
    getCalibrationFromFile,
    getMasterCameraMatrix,
)

testDataDir = Path(__file__).parent.parent / "data"


@pytest.fixture(scope="module")
def gesture_detector():
    """
    Fixture to load gesture detector. Only runs once per file.
    """
    g = Gesture()
    g.initialize()
    yield g
    g.finalize()


@pytest.fixture(
    params=[
        "raw_frame_1.png",
        "raw_frame_2.png",
        "raw_frame_3.png",
    ]
)
def test_file(request, test_data_dir):
    """
    Fixture to get test files.
    """
    file: Path = test_data_dir / request.param
    assert file.is_file(), "Test file does not exist"
    return file


def test_output(gesture_detector: Gesture, test_file):
    img = Image.open(test_file)

    file: Path = testDataDir / "calibration.json"
    assert file.is_file(), str(file) + " test file does not exist"
    skeletonJsonFile = open(file)
    skeletonData = json.load(skeletonJsonFile)
    _, rotation, translation, dist = getCalibrationFromFile(
        skeletonData["camera_calibration"]
    )

    img_interface = ColorImageInterface(frame_count=0, frame=np.asarray(img))
    depth = DepthImageInterface(frame_count=0, frame=np.zeros(shape=(640, 576)))
    cameraCalibration = CameraCalibrationInterface(
        cameraMatrix=getMasterCameraMatrix(),
        rotation=rotation,
        translation=translation,
        distortion=dist,
    )
    bodyTracking = BodyTrackingInterface(bodies=[], timestamp_usec=1000)
    output = gesture_detector.get_output(
        img_interface, depth, bodyTracking, cameraCalibration
    )

    assert isinstance(output, GestureConesInterface), str(output)
    assert (
        len(output.cones) > 0
    ), "No cones were identified, so there may be a problem with the model"

    # TODO check cone data similar to objects
    # for info in output.objects:
    #     assert len(info.center) == 3, str(info.center) + " center should be a 3d value"
    #     assert len(info.p1) == 2, "p1 should be a 2d value"
    #     assert len(info.p2) == 2, "p2 should be a 2d value"
    #     assert (
    #         info.p1[0] <= info.p2[0] and info.p1[1] <= info.p2[1]
    #     ), "the bottom right corner (p2) should have larger values than the top left corner (p1)"
    #     assert isinstance(info.object_class, GamrTarget), str(info.object_class)
