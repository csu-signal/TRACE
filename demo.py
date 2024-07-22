import os
from pathlib import Path

import cv2 as cv

from feature_manager import FeatureManager
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


    prof_7_22_run01 = create_recorded_profile(r"F:\brady_recording_tests\full_run_7_22\run01")
    prof_7_22_run02 = create_recorded_profile(r"F:\brady_recording_tests\full_run_7_22\run02")

    # prof: BaseProfile = live_prof
    prof: BaseProfile = LaptopProfile()
    # prof: BaseProfile = prof_7_19_run03

    output_directory = Path(prof.get_output_dir())
    processed_frame_dir = output_directory / "processed_frames"
    raw_frame_dir = output_directory / "raw_frames"

    os.makedirs(output_directory, exist_ok=False) # error if directory will get overwritten
    os.makedirs(processed_frame_dir, exist_ok=False)
    os.makedirs(raw_frame_dir, exist_ok=False)

    # initialize features
    features = FeatureManager(prof, output_dir = output_directory)

    # get device information
    device = prof.get_camera_device()
    device_id = 0
    cameraMatrix, rotation, translation, distortion = device.get_calibration_matrices()

    fail_count = 0
    frame_count = 0
    while True:
        # exit if 20 frames fail in a row
        if fail_count > 20:
            break

        # get output from camera or playback
        color_image, depth_image, body_frame_info = device.get_frame()
        if color_image is None or depth_image is None:
            print(f"DEVICE {device_id}: no color/depth image, skipping frame {frame_count}")
            frame_count += 1
            fail_count += 1
            continue
        else:
            fail_count = 0

        # process images into correct formats
        color_image = color_image[:,:,:3]
        depth = depth_image
        framergb = cv.cvtColor(color_image, cv.COLOR_BGR2RGB)
        output_frame = cv.cvtColor(color_image, cv.IMREAD_COLOR)
        cv.imwrite(f"{raw_frame_dir}\\frame{frame_count:08}.png", output_frame)

        bodies = body_frame_info["bodies"]

        # run all features
        features.processFrame(output_frame, framergb, depth, bodies, rotation, translation, cameraMatrix, distortion, frame_count)

        # resize, display, and save output
        output_frame = cv.resize(output_frame, (1280, 720))
        cv.imshow("output", output_frame)
        cv.waitKey(1)
        cv.imwrite(f"{processed_frame_dir}\\frame{frame_count:08}.png", output_frame)

        if cv.getWindowProperty("output", cv.WND_PROP_VISIBLE) == 0:
            break

        frame_count += 1

    # cleanup and generate processed video
    device.close()
    features.finalize()
    prof.finalize()
