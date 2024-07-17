from featureModules.asr.AsrFeature import AsrFeature
from featureModules.asr.device import MicDevice, PrerecordedDevice
from time import sleep
import numpy as np
import cv2 as cv


if __name__ == "__main__":
    asrPath = "asr_out.csv"
    # asr = AsrFeature([MicDevice('Participant 1',2),MicDevice('Participant 2',6),MicDevice('Participant 3',15)], n_processors=1, csv_log_file=asrPath)
    asr = AsrFeature([
        PrerecordedDevice('Videep',r'F:\brady_recording_tests\test_7_17-audio1-convert.wav', video_frame_rate=5),
        PrerecordedDevice('Austin',r'F:\brady_recording_tests\test_7_17-audio2-convert.wav',video_frame_rate=5),
        PrerecordedDevice('Mariah',r'F:\brady_recording_tests\test_7_17-audio3-convert.wav',video_frame_rate=5),
        ], n_processors=1, csv_log_file=asrPath)

    frame_count = 0
    while True:

        frame = np.zeros((720, 1280, 3))

        utterances = asr.processFrame(frame, frame_count)
        for id, start, stop, text, audio_file in utterances:
            print(f"{id}: {text}")


        cv.putText(frame, "FRAME:" + str(frame_count), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
        cv.imshow("frame", frame)
        cv.waitKey(30)
        if cv.getWindowProperty("frame", cv.WND_PROP_VISIBLE) == 0:
            break

        frame_count += 1

    asr.done.value = True
