import pytest
from mmdemo.features.gaze.gaze_feature import Gaze
from mmdemo.features.gesture.gesture_feature import Gesture
from mmdemo.features.objects.object_feature import Object
from mmdemo.features.output_frames.output_frames_feature import OutputFrames
from mmdemo.features.selected_objects.selected_objects_feature import SelectedObjects
from mmdemo.interfaces import CommonGroundInterface, GazeConesInterface, GestureConesInterface, ObjectInterface3D, SelectedObjectsInterface
from mmdemo.utils.hands import Handedness
import cv2 as cv


@pytest.fixture(scope="module")
def output_frames():
    """
    Fixture to load output frames detector. Only runs once per file.
    """
    o = OutputFrames()
    o.initialize()
    yield o
    o.finalize()

@pytest.fixture(scope="module")
def gesture_detector():
    """
    Fixture to load gesture detector. Only runs once per file.
    """
    o = Gesture()
    o.initialize()
    yield o
    o.finalize()

@pytest.fixture(scope="module")
def gaze_detector():
    """
    Fixture to load gaze detector. Only runs once per file.
    """
    o = Gaze()
    o.initialize()
    yield o
    o.finalize()

@pytest.fixture(scope="module")
def object_detector():
    """
    Fixture to load object detector. Only runs once per file.
    """
    o = Object()
    o.initialize()
    yield o
    o.finalize()

@pytest.fixture(scope="module")
def object_selector():
    """
    Fixture to load sekected object detector. Only runs once per file.
    """
    o = SelectedObjects()
    o.initialize()
    yield o
    o.finalize()

@pytest.mark.model_dependent
def test_output(
    output_frames: OutputFrames, 
    gesture_detector: Gesture, 
    gaze_detector: Gaze,
    object_selector: SelectedObjects, 
    azure_kinect_frame):

    color, depth, body_tracking, calibration = azure_kinect_frame
    gestureOutput = gesture_detector.get_output(color, depth, body_tracking, calibration)
    gazeOutput = gaze_detector.get_output(body_tracking, calibration)

    object_detector = Object(color, depth, calibration)
    objectOutput = object_detector.get_output(color, depth, calibration)

    selectedObjects = object_selector.get_output(objectOutput, gestureOutput.cones)
    common = CommonGroundInterface(fbank=[], ebank=[], qbank=[])

    output = output_frames.get_output(color, gazeOutput, gestureOutput, selectedObjects, common, calibration)
    cv.imshow("Output", output.frame)
    cv.waitKey(1)
