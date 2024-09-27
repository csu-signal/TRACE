import pytest

from mmdemo.features.gesture.gesture_feature import Gesture
from mmdemo.interfaces import GestureConesInterface
from mmdemo.interfaces.data import Handedness
from tests.utils.features import FakeFeature


@pytest.fixture(scope="module")
def gesture_detector():
    """
    Fixture to load gesture detector. Only runs once per file.
    """
    g = Gesture(FakeFeature(), FakeFeature(), FakeFeature(), FakeFeature())
    g.initialize()
    yield g
    g.finalize()


def test_default_model(gesture_detector):
    assert hasattr(
        gesture_detector, "DEFAULT_MODEL_PATH"
    ), "gesture should specify a default model path"
    assert (
        gesture_detector.DEFAULT_MODEL_PATH.is_file()
    ), "gesture model path should be an existing file"


@pytest.mark.model_dependent
def test_output(gesture_detector: Gesture, azure_kinect_frame, azure_kinect_frame_file):
    color, depth, body_tracking, calibration = azure_kinect_frame
    has_gesture = "gesture" in str(azure_kinect_frame_file)

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
