import cv2 as cv
import numpy as np
import pytest

from mmdemo.features.output_frames.emnlp_frame_feature import EMNLPFrame
from mmdemo.interfaces import (
    ColorImageInterface,
    CommonGroundInterface,
    GazeConesInterface,
    GestureConesInterface,
    SelectedObjectsInterface,
)
from mmdemo.interfaces.data import Cone, GamrTarget, Handedness, ObjectInfo3D
from tests.utils.fake_feature import FakeFeature


@pytest.fixture(scope="module")
def emnlp_frame():
    """
    Fixture to load output frame feature. Only runs once per file.
    """
    frame = EMNLPFrame(
        FakeFeature(), FakeFeature(), FakeFeature(), FakeFeature(), FakeFeature()
    )
    frame.initialize()
    yield frame
    frame.finalize()


@pytest.fixture
def gaze():
    return GazeConesInterface(
        cones=[
            Cone(
                base=np.array([22.0, -247.0, 1462.0]),
                vertex=np.array([-428.0, 214.0, 699.0]),
                base_radius=80,
                vertex_radius=100,
            ),
            Cone(
                base=np.array([406.0, -188.0, 981.0]),
                vertex=np.array([-463.0, 284.0, 836.0]),
                base_radius=80,
                vertex_radius=100,
            ),
            Cone(
                base=np.array([-456.0, -201.0, 1238.0]),
                vertex=np.array([155.0, 562.0, 1033.0]),
                base_radius=80,
                vertex_radius=100,
            ),
        ],
        body_ids=[3, 1, 2],
    )


@pytest.fixture
def gesture():
    return GestureConesInterface(
        cones=[
            Cone(
                base=np.array([-300.0, 0.0, 1500.0]),
                vertex=np.array([200.0, 100.0, 1600.0]),
                base_radius=40,
                vertex_radius=50,
            ),
        ],
        body_ids=[3],
        handedness=[Handedness.Left],
    )


@pytest.fixture
def objects():
    return SelectedObjectsInterface(
        objects=[
            (
                ObjectInfo3D(
                    p1=(940, 640),
                    p2=(960, 660),
                    object_class=GamrTarget.RED_BLOCK,
                    center=(0, 0, 0),
                ),
                True,
            ),
            (
                ObjectInfo3D(
                    p1=(750, 670),
                    p2=(780, 700),
                    object_class=GamrTarget.BLUE_BLOCK,
                    center=(0, 0, 0),
                ),
                False,
            ),
        ]
    )


@pytest.fixture
def common_ground():
    return CommonGroundInterface(
        qbank=set(),
        fbank={
            "red = 10",
            "red = 20",
            "red = 30",
            "red = 40",
            "red = 50",
            "blue = 10",
            "blue = 20",
            "green = 20",
        },
        ebank={"yellow = 50", "purple != 20"},
    )


def test_emnlp_frame(
    emnlp_frame, azure_kinect_frame, gaze, gesture, objects, common_ground
):
    color, _, _, calibration = azure_kinect_frame

    output = emnlp_frame.get_output(
        color, gaze, gesture, objects, common_ground, calibration
    )
    assert isinstance(output, ColorImageInterface)
    assert output.frame_count == color.frame_count, "Frame count should not change"
    assert (
        output.frame is not color.frame
    ), "Do not modify the color frame itself. This could break other features."

    resize = cv.resize(output.frame, (960, 540))
    cv.imshow("Output", resize)
    cv.waitKey(1000)
