import os
from pathlib import Path
from typing import final

import numpy as np
import pytest
from PIL import Image

from mmdemo.features.objects.object_feature import Object
from mmdemo.interfaces import CameraCalibrationInterface, ColorImageInterface, DepthImageInterface, ObjectInterface3D
from mmdemo.utils.Gamr import GamrTarget
from mmdemo.utils.camera_calibration_utils import getCalibrationFromFile, getMasterCameraMatrix
import json

from tests.utils.data import read_frame_pkl
from tests.utils.fake_feature import FakeFeature

testDataDir = Path(__file__).parent.parent / "data"

@pytest.fixture(scope="module")
def object_detector():
    """
    Fixture to load object detector. Only runs once per file.
    """
    o = Object(FakeFeature(), FakeFeature(), FakeFeature())
    o.initialize()
    yield o
    o.finalize()


@pytest.fixture(
    params=[
        "frame_01.pkl",
        "frame_02.pkl",
        "gesture_01.pkl",
        "gesture_01.pkl",
    ]
)
def test_data(request, test_data_dir):
    """
    Fixture to get test data.
    """
    file: Path = test_data_dir / request.param
    assert file.is_file(), "Test file does not exist"
    
    return read_frame_pkl(file)


def test_output(object_detector: Object, test_data):
    color, depth, _, calibration = test_data

    output = object_detector.get_output(color, depth, calibration)

    assert isinstance(output, ObjectInterface3D), str(output)
    assert (
        len(output.objects) > 0
    ), "No objects were identified, so there may be a problem with the model"

    for info in output.objects:
        assert len(info.center) == 3, str(info.center) + " center should be a 3d value"
        assert len(info.p1) == 2, "p1 should be a 2d value"
        assert len(info.p2) == 2, "p2 should be a 2d value"
        assert (
            info.p1[0] <= info.p2[0] and info.p1[1] <= info.p2[1]
        ), "the bottom right corner (p2) should have larger values than the top left corner (p1)"
        assert isinstance(info.object_class, GamrTarget), str(info.object_class)
