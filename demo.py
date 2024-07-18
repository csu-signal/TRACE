import os
from datetime import datetime
from dataclasses import dataclass

import cv2 as cv

from featureModules import (AsrFeature, BaseDevice, GazeBodyTrackingFeature,
                            GazeFeature, GestureFeature, MicDevice,
                            MoveFeature, ObjectFeature, PoseFeature,
                            PrerecordedDevice, PropExtractFeature, rec_common_ground)

from gui import Gui
from logger import Logger
from demo_profile import BaseProfile, LiveProfile, RecordedProfile


if __name__ == "__main__":
    live_prof = LiveProfile([
        ("Videep", 2),
        ("Austin", 6),
        ("Mariah", 15)
        ])

    recorded_prof = RecordedProfile(
        r"F:\brady_recording_tests\test_7_17-master.mkv",
        [
            ("Videep", r"F:\brady_recording_tests\test_7_17-audio1.wav"),
            ("Austin", r"F:\brady_recording_tests\test_7_17-audio2.wav"),
            ("Mariah", r"F:\brady_recording_tests\test_7_17-audio3.wav"),
        ])

    group1_prof = RecordedProfile(
        r"F:\Weights_Task\Data\Fib_weights_original_videos\Group_01-master.mkv",
        [
            ("Group 1", r"F:\Weights_Task\Data\Group_01-audio.wav"),
        ])

    prof: BaseProfile = recorded_prof

    gui = Gui()
    gui.create_buttons()

    output_directory = prof.get_output_dir()
    frame_dir = f"{output_directory}\\frames"
    gesturePath = f"{output_directory}\\gestureOutput.csv"
    objectPath = f"{output_directory}\\objectOutput.csv"
    posePath = f"{output_directory}\\poseOutput.csv"
    gazePath = f"{output_directory}\\gazeOutput.csv"
    asrPath = f"{output_directory}\\asrOutput.csv"
    propPath = f"{output_directory}\\propOutput.csv"
    movePath = f"{output_directory}\\moveOutput.txt"
    os.makedirs(output_directory, exist_ok=False) # error if directory will get overwritten
    os.makedirs(frame_dir, exist_ok=False)


    shift = 7 # TODO what is this?

    gaze = GazeBodyTrackingFeature(shift, csv_log_file=gazePath)
    gesture = GestureFeature(shift, csv_log_file=gesturePath)
    objects = ObjectFeature(csv_log_file=objectPath)
    pose = PoseFeature(csv_log_file=posePath)
    asr = AsrFeature(prof.get_audio_devices(), n_processors=1, csv_log_file=asrPath)
    prop = PropExtractFeature(csv_log_file=propPath)
    move = MoveFeature(txt_log_file=movePath)

    error_logger = Logger(file=f"{output_directory}\\errors.txt", stdout=True)
    error_logger.clear()

    device = prof.get_camera_device()
    device_id = 0
    cameraMatrix, rotation, translation, distortion = device.get_calibration_matrices()

    fail_count = 0
    frame_count = 0
    while True:
        # exit if 20 frames fail in a row
        if fail_count > 20:
            break

        gui.update()
        color_image, depth_image, body_frame_info = device.get_frame()
        if color_image is None or depth_image is None:
            print(f"DEVICE {device_id}: no color/depth image, skipping frame {frame_count}")
            frame_count += 1
            fail_count += 1
            continue
        else:
            fail_count = 0

        color_image = color_image[:,:,:3]
        depth = depth_image

        framergb = cv.cvtColor(color_image, cv.COLOR_BGR2RGB)
        frame = cv.cvtColor(color_image, cv.IMREAD_COLOR)

        h,w,_ = color_image.shape
        bodies = body_frame_info["bodies"]

        # run features
        blockStatus = {}
        blocks = []

        if(gui.should_process("objects")):
            blocks = objects.processFrame(framergb, frame_count, frame_count)

        if(gui.should_process("pose")):
            pose.processFrame(bodies, frame, frame_count)

        try:
            if(gui.should_process("gaze")):
                gaze.processFrame( bodies, w, h, rotation, translation, cameraMatrix, distortion, frame, framergb, depth, blocks, blockStatus, frame_count)
        except:
            pass
        
        if(gui.should_process("gesture")):
             gesture.processFrame(device_id, bodies, w, h, rotation, translation, cameraMatrix, distortion, frame, framergb, depth, blocks, blockStatus, frame_count, gesturePath)

        utterances = []
        if(gui.should_process("asr")):
            utterances = asr.processFrame(frame, frame_count)
            if(gui.should_process("gesture")):
                utterances = gesture.updateDemonstratives(utterances)

        utterances_and_props = []
        try:
            if(gui.should_process("prop")):
                utterances_and_props = prop.processFrame(frame, utterances, frame_count)
        except Exception as e:
            error_logger.append(f"Frame {frame_count}\nProp extractor\n{utterances}\n{str(e)}\n\n")

        if(gui.should_process("move")):
            move.processFrame(utterances_and_props, frame, frame_count)

        cv.putText(frame, "FRAME:" + str(frame_count), (50,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv.LINE_AA)
        cv.putText(frame, "DEVICE:" + str(int(device_id)), (50,100), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv.LINE_AA)

        frame = cv.resize(frame, (1280, 720))
        cv.imshow("output", frame)
        cv.waitKey(1)

        cv.imwrite(f"{frame_dir}\\frame{frame_count:08}.png", frame)

        if cv.getWindowProperty("output", cv.WND_PROP_VISIBLE) == 0:
            break

        frame_count += 1

    device.close()
    asr.done.value = True

    prof.finalize()
