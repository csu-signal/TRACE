import os###, sys###current_dir = os.path.dirname(os.path.abspath(__file__))###subdir_path = os.path.join(current_dir, 'subdir')###sys.path.append(current_dir)
from demo.config import K4A_DIR, PLAYBACK_SKIP_FRAMES, WTD_END_TIMES
from demo.featureModules.gaze.GazeBodyTrackingFeature import GazeBodyTrackingFeature

os.add_dll_directory(K4A_DIR)
from azure_kinect import Playback
import cv2 as cv

from demo.profiles import (
    ###BradyLaptopProfile,
    ###LiveProfile,
    RecordedProfile,
    ###create_recorded_profile,
    ###create_wtd_eval_profiles,
)
from demo.base_profile import BaseProfile, FrameInfo

###from demo.featureModules import rec_common_ground
###added below
import datetime
import csv

if __name__ == "__main__":
    print("__main__: past from azure_kinect import Playback")
    print(datetime.datetime.now(),"\n")
    # live_prof = LiveProfile([
    #     ("Videep", 2),
    #     ("Austin", 6),
    #     ("Mariah", 15)
    #     ])

    # live_prof = LiveProfile([
    #     ("Group", 6),
    #     ])

    groups = [1, 2, 4, 5]

    profiles: list[BaseProfile] = []
    '''
    for group in groups:
        profiles += create_wtd_eval_profiles(
            group, "wtd_inputs", "wtd_outputs", end_time=WTD_END_TIMES[group]
        )
    '''
    ###Replace setting profiles above with this next line for a single pre-recorded file. Need example of path: in create_recorded_profile it's rf"{path}-master.mkv"; r is raw, which means backslashes are
    ###treated as \ not as an escape char, and f is formatted, so the {path} is replaced by the value of path.
    path = r"C:\Users\jimmu\Desktop\CS793\mkv files\run02-master.mkv"
    ###profiles.append(RecordedProfile(rf"{path}-master.mkv",[]))###, output_dir=output_dir, eval_config=eval_config)
    profiles.append(RecordedProfile(path,[]))
    
    ###this loop is to input a Weights Task Dataset/Group_xx/Group_xx-master-skeleton.json. Either run this loop
    ###or the following loop, but not both in the same run.
    for prof in profiles:
        prof.init_features()
        # get device information
        device = prof.create_camera_device()
        cameraMatrix, rotation, translation, distortion = (
            device.get_calibration_matrices()
        )
        print("cameraMatrix\n",cameraMatrix,"\ndistortion\n",distortion)
        print(cameraMatrix.dtype, distortion.dtype)
        master_skeleton_dir = r"C:\Users\jimmu\Desktop\CS793\Weights Task Dataset"
        ###for i in range(1,11):
        ###for i in range(1,10):
        for i in range(1,11): ###loop through all master-skeleton.json files for gropus 1 - 10
            extension = '0'+str(i)
            if i == 10:
                extension = '10'
            master_skeleton_file = master_skeleton_dir+'\\Group_'+extension+'\\Group_'+extension+'-master-skeleton.json'
            collect_joint_coords_from_json_file = GazeBodyTrackingFeature(7) ###no idea what the 7 is for, it's in base_profile.py
            ###this next call actually generates the input file based on the Group01 - Group10 data
            collect_joint_coords_from_json_file.processMasterSkeleton(rotation, translation, cameraMatrix, 
                                                                      distortion, master_skeleton_file)
        device.close()
    '''This loop is for reading from a pre-recorded .mkv file
    for prof in profiles:
        prof.init_features()
        # get device information
        device = prof.create_camera_device()
        cameraMatrix, rotation, translation, distortion = (
            device.get_calibration_matrices()
        )

        fail_count = 0
        print("__main__: frame_count",end=".")
        while True:
            # skip frames to match target fps
            ###print("__main__: isinstance(device,Playback)", isinstance(device, Playback))
            if isinstance(device, Playback):
                device.skip_frames(PLAYBACK_SKIP_FRAMES)

            # get output from camera or playback
            color_image, depth_image, body_frame_info = device.get_frame() ###get_frame is line 82 in device.cpp
            if color_image is None or depth_image is None:
                print(f"no color/depth image, skipping frame")
                fail_count += 1
                continue

            fail_count = 0

            color_image = color_image[:, :, :3]
            frame_count = device.get_frame_count()
    '''
    '''###keep this if running second for prof in profiles loop
            @dataclass
            class FrameInfo:
                output_frame: MatLike
                framergb: MatLike
                depth: MatLike
                bodies: list
                rotation: np.ndarray
                translation: np.ndarray
                cameraMatrix: np.ndarray
                distortion: np.ndarray
                frame_count: int
    '''###keep this if running second for prof in profiles loop
    '''
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
            ###prof.preprocess(frame_info)
            prof.processFrame(frame_info)
            ###prof.postprocess(frame_info)
            ###print("sleep for three sec")
            ###time.sleep(3)
            ###quit() ###for testing
            # exit if the profile is completed
            print(frame_count,end=".")
    '''
    '''keep this if running second for prof in profiles loop
            if frame_count < 10:
                with open('depth_info.csv','w') as f:
                    writer = csv.writer(f)
                    writer.writerows(frame_info.depth[:][:])
    '''###keep this if running second for prof in profiles loop
    '''
            ###if prof.is_done(frame_count, fail_count):
            if frame_count > 4200:
                print("__main__: fail_count",fail_count)
                break
        print("__main__: frame_count",frame_count)
        # cleanup and generate processed video
        device.close()
        ###prof.finalize()
    '''
