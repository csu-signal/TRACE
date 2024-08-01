"""
Test ASR without running the full demo.
"""

from demo.base_profile import FrameTimeConverter
from demo.featureModules.asr.AsrFeature import AsrFeature
from demo.featureModules.asr.device import MicDevice, PrerecordedDevice
from time import time
import numpy as np
import cv2 as cv


if __name__ == "__main__":
    asr = AsrFeature([
        MicDevice("User", 1)
        ], n_processors=1, log_dir=None)

    time_to_frame = FrameTimeConverter()

    frame_count = 0
    while True:

        time_to_frame.add_data(frame_count, time())

        frame = np.zeros((720, 1280, 3))
        new_utterances = asr.processFrame(frame, frame_count, time_to_frame.get_frame, False)
        for id in new_utterances:
            utterance_info = asr.utterance_lookup[id]
            print(f"{utterance_info.utterance_id}: {utterance_info.text}")

        cv.putText(frame, "FRAME:" + str(frame_count), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
        cv.imshow("frame", frame)
        cv.waitKey(30)
        if cv.getWindowProperty("frame", cv.WND_PROP_VISIBLE) == 0:
            break

        frame_count += 1

    asr.done.value = True
