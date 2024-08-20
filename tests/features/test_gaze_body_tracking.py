import numpy as np
import pytest

from mmdemo.features.gaze.gaze_feature import Gaze
from mmdemo.interfaces import BodyTrackingInterface, GazeConesInterface
from mmdemo.interfaces.data import Cone
from mmdemo.utils.coordinates import world_3d_to_camera_3d
from mmdemo.utils.joints import Joint


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
    base=np.array([-201.3333, -81.3333, 1.5]),
    vertex=np.array([353.366, 750.716, 1.5]),
    base_radius=80,
    vertex_radius=100,
)

p2 = create_body_dict(
    2, [2, -70, 1.7], [0, -80, 1.75], [4, -80, 1.75], [-2, -75, 1.9], [6, -75, 1.9]
)
p2_expected = Cone(
    base=np.array([2, -76.6666666, 1.733333]),
    vertex=np.array([2, 922.5342, -38.235]),
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
    base=np.array([191.33333, -86.33333, 1.3]),
    vertex=np.array([-363.36654, 745.71648, 1.3]),
    base_radius=80,
    vertex_radius=100,
)


def get_dir(base, vertex):
    v = np.array(vertex) - np.array(base)
    return v / np.linalg.norm(v)


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
def test_gaze_body_tracking_formula(gaze, camera_calibration, bodies, expected_output):
    """
    Test that gazes use the correct formula. This should be:
    - origin = average of nose and eyes
    - dir = vector between average of ears and the nose
    """
    body_tracking_interface = BodyTrackingInterface(bodies=bodies, timestamp_usec=-1)
    output = gaze.get_output(body_tracking_interface, camera_calibration)
    assert isinstance(output, GazeConesInterface)

    for cone, expected in zip(output.cones, expected_output):
        # expected is in world coords while the output should be in camera coords
        expected_base = world_3d_to_camera_3d(expected.base, camera_calibration)
        expected_vertex = world_3d_to_camera_3d(expected.vertex, camera_calibration)

        assert np.allclose(
            cone.base, expected_base
        ), "Gaze does not start from the correct place"
        assert np.allclose(
            get_dir(cone.base, cone.vertex),
            get_dir(expected_base, expected_vertex),
        ), "Gaze does not point in the correct direction"
