import os
from config import K4A_DIR, PLAYBACK_SKIP_FRAMES

os.add_dll_directory(K4A_DIR)
from azure_kinect import Playback
import cv2 as cv

from profiles import BradyLaptopProfile, LiveProfile, RecordedProfile, create_recorded_profile, TestDenseParaphrasingProfile
from base_profile import BaseProfile

from featureModules import rec_common_ground


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


    # prof_7_24_run01 = create_recorded_profile(r"F:\brady_recording_tests\full_run_7_24\run01")
    # prof_7_24_run02 = create_recorded_profile(r"F:\brady_recording_tests\full_run_7_24\run02")
    # prof_7_24_run01_gt = create_recorded_profile(r"F:\brady_recording_tests\full_run_7_24\run01", eval_dir="run01_gt_input", eval_asr=True, eval_prop=True, eval_move=True, eval_gesture=True)

    prof: BaseProfile = TestDenseParaphrasingProfile()
    prof.init_features()


    # get device information
    device = prof.create_camera_device()
    device_id = 0
    cameraMatrix, rotation, translation, distortion = device.get_calibration_matrices()

    fail_count = 0
    frame_count = 0
    saved_frame_count = 0
    while True:
        # exit if 20 frames fail in a row
        if fail_count > 20:
            break

        # skip frames to match target fps
        if isinstance(device, Playback):
            device.skip_frames(PLAYBACK_SKIP_FRAMES)
            frame_count += PLAYBACK_SKIP_FRAMES

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
        cv.imwrite(f"{prof.raw_frame_dir}\\frame{saved_frame_count:08}.png", output_frame)

        bodies = body_frame_info["bodies"]

        # run all features
        prof.processFrame(output_frame, framergb, depth, bodies, rotation, translation, cameraMatrix, distortion, frame_count)

        # resize, display, and save output
        output_frame = cv.resize(output_frame, (1280, 720))
        cv.imshow("output", output_frame)
        cv.waitKey(1)
        cv.imwrite(f"{prof.processed_frame_dir}\\frame{saved_frame_count:08}.png", output_frame)

        if cv.getWindowProperty("output", cv.WND_PROP_VISIBLE) == 0:
            break

        frame_count += 1
        saved_frame_count += 1

    # cleanup and generate processed video
    device.close()
    prof.finalize()
