import os
from pathlib import Path

import cv2 as cv

from featureModules import (AsrFeature, BaseDevice, GazeBodyTrackingFeature,
                            GazeFeature, GestureFeature, MicDevice,
                            MoveFeature, ObjectFeature, PoseFeature,
                            PrerecordedDevice, PropExtractFeature, rec_common_ground, DenseParaphrasingFeature, CommonGroundFeature)

from gui import Gui
from logger import Logger
from input_profile import BaseProfile, LaptopProfile, LiveProfile, RecordedProfile, create_recorded_profile


if __name__ == "__main__":
    # live_prof = LiveProfile([
    #     ("Videep", 2),
    #     ("Austin", 6),
    #     ("Mariah", 15)
    #     ])

    live_prof = LiveProfile([
        ("Group", 6),
        ])


    group1 = RecordedProfile(
        r"F:\Weights_Task\Data\Fib_weights_original_videos\Group_01-master.mkv",
        [
            ("Group 1", r"F:\Weights_Task\Data\Group_01-audio.wav"),
        ])


    prof_7_18_run02 = create_recorded_profile(r"F:\brady_recording_tests\full_run_7_18\run02")
    prof_7_19_run02 = create_recorded_profile(r"C:\Users\brady03\Desktop\full_run_7_19\run02")
    prof_7_19_run03 = create_recorded_profile(r"C:\Users\brady03\Desktop\full_run_7_19\run03")

    # prof: BaseProfile = live_prof
    prof: BaseProfile = LaptopProfile()
    # prof: BaseProfile = prof_7_19_run03

    gui = Gui()
    gui.create_buttons()

    output_directory = Path(prof.get_output_dir())
    processed_frame_dir = output_directory / "processed_frames"
    raw_frame_dir = output_directory / "raw_frames"

    os.makedirs(output_directory, exist_ok=False) # error if directory will get overwritten
    os.makedirs(processed_frame_dir, exist_ok=False)
    os.makedirs(raw_frame_dir, exist_ok=False)


    shift = 7 # TODO what is this?

    gaze = GazeBodyTrackingFeature(shift, log_dir=output_directory)
    gesture = GestureFeature(shift, log_dir=output_directory)
    objects = ObjectFeature(log_dir=output_directory)
    pose = PoseFeature(log_dir=output_directory)
    asr = AsrFeature(prof.get_audio_devices(), n_processors=1, log_dir=output_directory)
    dense_paraphrasing = DenseParaphrasingFeature(log_dir=output_directory)
    prop = PropExtractFeature(log_dir=output_directory)
    move = MoveFeature(log_dir=output_directory)
    common_ground = CommonGroundFeature(log_dir=output_directory)

    error_log = Logger(file=output_directory / "errors.txt", stdout=True)
    error_log.clear()

    summary_log = Logger(file=output_directory / "summary.txt", stdout=True)
    summary_log.clear()

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
        cv.imwrite(f"{raw_frame_dir}\\frame{frame_count:08}.png", frame)

        h,w,_ = color_image.shape
        bodies = body_frame_info["bodies"]

        # run features
        blockStatus = {}
        blocks = []

        if(gui.should_process("objects")):
            blocks = objects.processFrame(framergb, frame_count)

        if(gui.should_process("pose")):
            pose.processFrame(bodies, frame, frame_count, False)

        try:
            if(gui.should_process("gaze")):
                gaze.processFrame( bodies, w, h, rotation, translation, cameraMatrix, distortion, frame, framergb, depth, blocks, blockStatus, frame_count)
        except:
            pass
        
        if(gui.should_process("gesture")):
             gesture.processFrame(device_id, bodies, w, h, rotation, translation, cameraMatrix, distortion, frame, framergb, depth, blocks, blockStatus, frame_count, False)

        new_utterances = []
        if(gui.should_process("asr")):
            new_utterances = asr.processFrame(frame, frame_count, False)

        if gui.should_process("dense paraphrasing"):
            dense_paraphrasing.processFrame(frame, new_utterances, asr.utterance_lookup, gesture.blockCache, frame_count)

        try:
            if(gui.should_process("prop")):
                prop.processFrame(frame, new_utterances, dense_paraphrasing.paraphrased_utterance_lookup, frame_count, False)

        except Exception as e:
            error_log.append(f"Frame {frame_count}\nProp extractor\n{new_utterances}\n{str(e)}\n\n")

        if(gui.should_process("move")):
            move.processFrame(frame, new_utterances, dense_paraphrasing.paraphrased_utterance_lookup, frame_count, False)

        if gui.should_process("common ground"):
            common_ground.processFrame(frame, new_utterances, prop.prop_lookup, move.move_lookup, frame_count)


        cv.putText(frame, "FRAME:" + str(frame_count), (50,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv.LINE_AA)
        #cv.putText(frame, "DEVICE:" + str(int(device_id)), (50,100), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv.LINE_AA)

        # update = ""
        # update += "FRAME: " + str(frame_count) + "\n"
        # update += "Q bank\n"
        # update += str(self.closure_rules.qbank) + "\n"
        # update += "E bank\n"
        # update += str(self.closure_rules.ebank) + "\n"
        # update += "F bank\n"
        # update += str(self.closure_rules.fbank) + "\n"
        # if prop == "no prop":
        #     update += f"{name}: {text} ({self.most_recent_prop}), {out}\n\n"
        # else:
        #     update += f"{name}: {text} => {self.most_recent_prop}, {out}\n\n"

        frame = cv.resize(frame, (1280, 720))
        cv.imshow("output", frame)
        cv.waitKey(1)

        cv.imwrite(f"{processed_frame_dir}\\frame{frame_count:08}.png", frame)

        if cv.getWindowProperty("output", cv.WND_PROP_VISIBLE) == 0:
            break

        frame_count += 1

    device.close()
    asr.done.value = True

    prof.finalize()
