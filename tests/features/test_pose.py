import pytest

from mmdemo.features.pose.pose_feature import Pose
from mmdemo.interfaces import GestureConesInterface, PoseInterface
from mmdemo.interfaces.data import Handedness
from tests.utils.fake_feature import FakeFeature


@pytest.fixture(scope="module")
def pose_detector():
    """
    Fixture to load pose detector. Only runs once per file.
    """
    p = Pose(FakeFeature())
    p.initialize(-400, 400)
    yield p
    p.finalize()


def test_output(pose_detector: Pose, azure_kinect_frame):
    _, _, body_tracking, _ = azure_kinect_frame
    output = pose_detector.get_output(body_tracking)

    assert isinstance(output, PoseInterface), str(output)
    assert len(output.poses) == 3
