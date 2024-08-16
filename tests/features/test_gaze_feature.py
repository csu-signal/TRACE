from typing import final

import numpy as np
import pytest

from mmdemo.features.gaze.gaze_feature import Gaze
from mmdemo.interfaces import (
    BodyTrackingInterface,
    CameraCalibrationInterface,
    GazeConesInterface,
)
from mmdemo.interfaces.data import Cone
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


def create_body_dict(id, nose, left_eye, right_eye, left_ear, right_ear):
    """
    Helper function to create body tracking interfaces
    """
    return {
        "body_id": id,
        "joint_positions": {
            Joint.NOSE.value: nose,
            Joint.EYE_LEFT.value: left_eye,
            Joint.EAR_LEFT.value: left_ear,
            Joint.EYE_RIGHT.value: right_eye,
            Joint.EAR_RIGHT.value: right_ear,
        },
        "joint_orientation": [],
    }


# TODO: make 3 example inputs and expected outputs. also I think our
# Vectors3DInterface is not actually correct for this feature -- gaze needs to
# output both a direction and origin, so you will probably need to modify it in order to
# make that possible. Also either let Hannah know how you are modifying it or work together
# with her to do it since she will also need to do the same thing for gestures

# The coordinate system is oriented such that the positive X-axis points right,
# the positive Y-axis points down, and the positive Z-axis points forward.

# body_id, nose, l.eye, r.eye, l.ear, r.ear
p1 = create_body_dict(
    1,
    [-200, -75, 1.5],
    [-201, -84, 1.45],
    [-203, -85, 1.55],
    [-202, -79, 1.3],
    [-204, -80, 1.7],
)
p1_expected = Cone(
    base=[-201.33333333333333333333333333333, -81.333333333333333333333333333333, 1.5],
    vertex=[353.366546267, 750.71648567, 1.5],
    base_radius=80,
    vertex_radius=100,
)

p2 = create_body_dict(
    2, [2, -70, 1.7], [0, -80, 1.75], [4, -80, 1.75], [-2, -75, 1.9], [6, -75, 1.9]
)
p2_expected = Cone(
    base=[2, -76.666666666666666666666666666667, 1.7333333333333333333333333333333],
    vertex=[2, 1310.08382389, -3.81366862667],
    base_radius=80,
    vertex_radius=100,
)

p3 = create_body_dict(
    3,
    [190, -80, 1.3],
    [193, -90, 1.25],
    [191, -89, 1.35],
    [194, -85, 1.5],
    [192, -84, 1.1],
)
p3_expected = Cone(
    base=[191.33333333333333333333333333333, -86.333333333333333333333333333333, 1.3],
    vertex=[-363.36654629678666666666666666667, 745.71648566666666666666666666667, 1.3],
    base_radius=80,
    vertex_radius=100,
)


# this is the test that will run with pytest, we parameterize
# joints and expected_output to run the test with different inputs
@pytest.mark.parametrize(
    "bodies,expected_output",
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
    assert isinstance(output, GazeConesInterface)

    for cone, expected in zip(output.cones, expected_output):
        assert np.allclose(cone.vertex, expected.vertex)
        assert np.allclose(cone.base, expected.base)
        assert np.allclose(cone.vertex_radius, expected.vertex_radius)
        assert np.allclose(cone.base_radius, expected.base_radius)
