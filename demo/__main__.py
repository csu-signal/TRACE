import os
from demo.config import K4A_DIR, PLAYBACK_SKIP_FRAMES, WTD_END_TIMES

os.add_dll_directory(K4A_DIR)
from azure_kinect import Playback
import cv2 as cv

from demo.profiles import (
    BradyLaptopProfile,
    LiveProfile,
    RecordedProfile,
    create_recorded_profile,
    create_wtd_eval_profiles,
)
from demo.base_profile import BaseProfile, FrameInfo

from demo.featureModules import rec_common_ground


if __name__ == "__main__":
    # live_prof = LiveProfile([
    #     ("Videep", 2),
    #     ("Austin", 6),
    #     ("Mariah", 15)
    #     ])

    # live_prof = LiveProfile([
    #     ("Group", 6),
    #     ])

    # groups = [1, 2, 4, 5]
    #
    # profiles: list[BaseProfile] = []
    # for group in groups:
    #     profiles += create_wtd_eval_profiles(
    #         group, "wtd_inputs", "wtd_outputs", end_time=WTD_END_TIMES[group]
    #     )

    profiles = [BradyLaptopProfile()]
    for prof in profiles:
        prof.init_features()

        # get device information
        device = prof.create_camera_device()
        cameraMatrix, rotation, translation, distortion = (
            device.get_calibration_matrices()
        )

        fail_count = 0
        while True:
            # skip frames to match target fps
            if isinstance(device, Playback):
                device.skip_frames(PLAYBACK_SKIP_FRAMES)

            # get output from camera or playback
            color_image, depth_image, body_frame_info = device.get_frame()
            if color_image is None or depth_image is None:
                print(f"no color/depth image, skipping frame")
                fail_count += 1
                continue

            fail_count = 0

            color_image = color_image[:, :, :3]
            frame_count = device.get_frame_count()

            frame_info = FrameInfo(
                output_frame=cv.cvtColor(color_image, cv.IMREAD_COLOR),
                framergb=cv.cvtColor(color_image, cv.COLOR_BGR2RGB),
                depth=depth_image,
                bodies=body_frame_info["bodies"],
                rotation=rotation,
                translation=translation,
                cameraMatrix=cameraMatrix,
                distortion=distortion,
                frame_count=frame_count,
            )

            # process frame
            prof.preprocess(frame_info)
            prof.processFrame(frame_info)
            prof.postprocess(frame_info)

            # exit if the profile is completed
            if prof.is_done(frame_count, fail_count):
                break

        # cleanup and generate processed video
        device.close()
        prof.finalize()
