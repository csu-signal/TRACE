#TODO stub out demo code for HCII Individual Tracking Paper

from mmdemo.features.gaze.gaze_body_tracking_feature import GazeBodyTracking
from mmdemo.features.gesture.gesture_feature import Gesture
from mmdemo.features.gesture.hcii_gesture_feature import HciiGesture
from mmdemo.features.outputs.display_frame_feature import DisplayFrame
from mmdemo.features.outputs.hcii_it_frame_feature import HCII_IT_Frame
from mmdemo.features.outputs.hcii_logging_feature import HciiLog
from mmdemo.features.outputs.logging_feature import Log
from mmdemo.features.outputs.save_video_feature import SaveVideo
from mmdemo_azure_kinect import DeviceType, create_azure_kinect_features
from pathlib import Path
from mmdemo.demo import Demo

# mkv path for WTD group
WTD_MKV_PATH = (
    "G:/Weights_Task/Data/Fib_weights_original_videos/Group_{0:02}-master.mkv"
)

# Number of frames to evaluate per second. This must
# be a divisor of 30 (the true frame rate). Higher rates
# will take longer to process.
PLAYBACK_FRAME_RATE = 30

# The number of seconds of the recording to process
WTD_END_TIMES = {
    1: 5 * 60 + 30,
    2: 5 * 60 + 48,
    3: 8 * 60 + 3,
    4: 3 * 60 + 31,
    5: 4 * 60 + 34,
    6: 5 * 60 + 3,
    7: 8 * 60 + 30,
    8: 6 * 60 + 28,
    9: 3 * 60 + 46,
    10: 6 * 60 + 51,
    11: 2 * 60 + 19,
}

if __name__ == "__main__":

     # load azure kinect features from file
    group = 1
    color, depth, body_tracking, calibration = create_azure_kinect_features(
        DeviceType.PLAYBACK,
        mkv_path=Path(WTD_MKV_PATH.format(group)),
        playback_end_seconds=WTD_END_TIMES[group],
        playback_frame_rate=PLAYBACK_FRAME_RATE,
    )

     # gaze and gesture
    gaze = GazeBodyTracking(body_tracking, calibration)
    gesture = HciiGesture(color, depth, body_tracking, calibration)

    # create output frame for video
    output_frame = HCII_IT_Frame(color, gaze, gesture, calibration)

     # run demo and show output
    demo = Demo(
        targets=[
            DisplayFrame(output_frame),
            #SaveVideo(output_frame),
            HciiLog(gesture, fileName="IndividualTracking_Group1.csv", csv=True)
            #Log(body_tracking, csv=True), #TODO log individual tracking values
        ]
    )
    demo.show_dependency_graph()
    demo.run()