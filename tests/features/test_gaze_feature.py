from typing import final

import numpy as np
import pytest

from mmdemo.features.gaze.gaze_feature import Gaze
from mmdemo.interfaces import (
    BodyTrackingInterface,
    CameraCalibrationInterface,
    Vectors3DInterface,
)
from mmdemo.utils.support_utils import Joint


# fixtures are data that we want to use in the test.
# by default they are automatically recreated every
# time they are used, but we can specify `scope="module"`
# to only initialize once per file. this is helpful for
# features that may need to load an expensive model and
# do not store internal state
@pytest.fixture(scope="module")
def gaze():
    g = Gaze()
    g.initialize()
    yield g
    g.finalize()


# here we use a fixture to get an example camera calibration
@pytest.fixture
def cc_interface():
    return CameraCalibrationInterface(
        rotation=np.eye(3),
        translation=np.zeros(3),
        cameraMatrix=np.array([]),
        distortion=np.array([]),
    )


def create_body_dict(nose, eye_avg, ear_avg):
    """
    Helper function to create body tracking interfaces
    """
    # TODO: create a bodies dict that has its nose joint at `nose`,
    # average of eye joints at `eye_avg`, and average of ear joints at `ear_avg`
    # refer to gaze feature code to figure out how to do this
    bodies = ...
    return BodyTrackingInterface(bodies=bodies, timestamp_usec=-1)


# TODO: make 3 example inputs and expected outputs. also I think our
# Vectors3DInterface is not actually correct for this feature -- gaze needs to
# output both a direction and origin, so you will probably need to modify it in order to
# make that possible. Also either let Hannah know how you are modifying it or work together
# with her to do it since she will also need to do the same thing for gestures
p1 = create_body_dict([0, 0, 0], [1, 0, 0], [0, 1, 0])
p1_expected = [0, 0, 0]

p2 = create_body_dict([0, 0, 0], [1, 0, 0], [0, 1, 0])
p2_expected = [0, 0, 0]

p3 = create_body_dict([0, 0, 0], [1, 0, 0], [0, 1, 0])
p3_expected = [0, 0, 0]


# this is the test that will run with pytest, we parameterize
# joints and expected_output to run the test with different inputs
@pytest.mark.parametrize(
    "bodies,expected_outputs",
    [
        ([p1], [p1_expected]),
        ([p2], [p2_expected]),
        ([p3], [p3_expected]),
        ([p1, p2, p3], [p1_expected, p2_expected, p3_expected]),
        ([p3, p1, p2], [p3_expected, p1_expected, p2_expected]),
    ],
)
def test_gaze_body_tracking_formula(gaze, cc_interface, bodies, expected_output):
    body_tracking_interface = BodyTrackingInterface(bodies=bodies, timestamp_usec=-1)
    output = gaze.get_output(body_tracking_interface, cc_interface)
    assert isinstance(output, Vectors3DInterface)

    for out, expected in zip(output.vectors, expected_output):
        assert np.isclose(out, expected)
