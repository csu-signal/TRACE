from pathlib import Path

import pytest

from mmdemo.features.gesture.gesture_feature import Gesture
from mmdemo.interfaces import GestureConesInterface
from mmdemo.utils.hands import Handedness
from tests.utils.data import read_frame_pkl

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
        "gesture_01.pkl",
        "gesture_02.pkl",
        "frame_01.pkl",
        "frame_02.pkl",
    ]
)
def test_data(request, test_data_dir):
    """
    Fixture to get test files. Files with gesture
    in the name should contain a gesture and other
    files should not.
    """
    file: Path = test_data_dir / request.param
    assert file.is_file(), "Test file does not exist"
    return read_frame_pkl(file), "gesture" in request.param


@pytest.mark.model_dependent
def test_output(gesture_detector: Gesture, test_data):
    (color, depth, body_tracking, calibration), has_gesture = test_data

    output = gesture_detector.get_output(color, depth, body_tracking, calibration)

    assert isinstance(output, GestureConesInterface), str(output)
    if has_gesture:
        assert len(output.cones) == 1, "This test input should have one gesture"
    else:
        assert len(output.cones) == 0, "This test input should have no gestures"

    assert len(output.cones) == len(output.handedness)
    assert len(output.cones) == len(output.body_ids)

    for cone, body_id, handedness in zip(
        output.cones, output.body_ids, output.handedness
    ):
        assert cone.base.shape == (3,), "base should be 3d np array"
        assert cone.vertex.shape == (3,), "vertex should be 3d np array"

        assert isinstance(body_id, int)
        assert isinstance(handedness, Handedness)
