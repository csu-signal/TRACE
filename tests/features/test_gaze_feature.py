from typing import final

import pytest

from mmdemo.features.gaze.gaze_feature import Gaze
from mmdemo.interfaces import (
    BodyTrackingInterface,
    CameraCalibrationInterface,
    Vectors3DInterface,
)


# TODO: Incomplete test file
@final
@pytest.mark.xfail
def test_import():
    """
    Check that imports work
    """
    from mmdemo.interfaces import (
        BodyTrackingInterface,
        CameraCalibrationInterface,
        Vectors3DInterface,
    )


@final
@pytest.mark.xfail
def test_input_interfaces(gaze: Gaze):
    args = gaze.get_input_interfaces()
    assert len(args) == 2
    assert isinstance(args, list)


@pytest.mark.xfail
def test_output_interface(gaze: Gaze):
    assert isinstance(gaze.get_output_interface(), Vectors3DInterface)


def test_output(gaze: Gaze):
    output = gaze.get_output()
    # assert isinstance(output, Vectors3DInterface)
    assert len(output.vectors) == 2
    assert isinstance(output.vectors[0], tuple)
    assert isinstance(output.vectors[1], tuple)


gaze = Gaze()

test_import()
test_input_interfaces(gaze)
test_output_interface(gaze)
test_output(gaze)
