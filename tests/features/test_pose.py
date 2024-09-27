import pytest

from mmdemo.features.pose.pose_feature import Pose
from mmdemo.interfaces import PoseInterface
from tests.utils.features import FakeFeature


@pytest.fixture(scope="module")
def pose_detector():
    """
    Fixture to load pose detector. Only runs once per file.
    """
    p = Pose(FakeFeature())
    p.initialize()
    yield p
    p.finalize()


def test_default_model(pose_detector):
    assert hasattr(
        pose_detector, "DEFAULT_MODEL_PATH"
    ), "pose should specify a default model path"
    assert (
        pose_detector.DEFAULT_MODEL_PATH.is_dir()
    ), "pose model path should be an existing directory"


@pytest.mark.model_dependent
def test_output(pose_detector: Pose, azure_kinect_frame):
    _, _, body_tracking, _ = azure_kinect_frame
    output = pose_detector.get_output(body_tracking)

    assert isinstance(output, PoseInterface), str(output)
    assert len(output.poses) == 3

    for _, pose in output.poses:
        assert pose in ["leaning out", "leaning in"]
