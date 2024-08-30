import pytest

from mmdemo.features.objects.object_feature import Object
from mmdemo.interfaces import ObjectInterface3D
from mmdemo.interfaces.data import GamrTarget
from tests.utils.features import FakeFeature


@pytest.fixture(scope="module")
def object_detector():
    """
    Fixture to load object detector. Only runs once per file.
    """
    o = Object(FakeFeature(), FakeFeature(), FakeFeature())
    o.initialize()
    yield o
    o.finalize()


def test_default_model(object_detector):
    assert hasattr(
        object_detector, "DEFAULT_MODEL_PATH"
    ), "object should specify a default model path"
    assert (
        object_detector.DEFAULT_MODEL_PATH.is_file()
    ), "object model path should be an existing file"


@pytest.mark.model_dependent
def test_output(object_detector: Object, azure_kinect_frame):
    color, depth, _, calibration = azure_kinect_frame

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
