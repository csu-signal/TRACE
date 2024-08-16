import os
from pathlib import Path
from typing import final

import numpy as np
import pytest
from PIL import Image

from mmdemo.features.objects.object_feature import Object
from mmdemo.interfaces import ColorImageInterface, ObjectInterface3D
from mmdemo.utils.Gamr import GamrTarget


@pytest.fixture(scope="module")
def object_detector():
    """
    Fixture to load object detector. Only runs once per file.
    """
    o = Object()
    o.initialize()
    yield o
    o.finalize()


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


def test_output(object_detector: Object, test_file):
    img = Image.open(test_file)
    img_interface = ColorImageInterface(frame_count=0, frame=np.asarray(img))
    output = object_detector.get_output(img_interface)

    assert isinstance(output, ObjectInterface3D)
    assert (
        len(output.objects) > 0
    ), "No objects were identified, so there may be a problem with the model"

    for info in output.objects:
        assert len(info.center) == 3, "Center should be a 3d value"
        assert len(info.p1) == 2, "p1 should be a 2d value"
        assert len(info.p2) == 2, "p2 should be a 2d value"
        assert (
            info.p1[0] <= info.p2[0] and info.p1[1] <= info.p2[1]
        ), "the bottom right corner (p2) should have larger values than the top left corner (p1)"
        assert isinstance(info.object_class, GamrTarget)