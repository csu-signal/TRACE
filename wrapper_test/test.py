import os

import numpy as np
import cv2 as cv

from featureModules.gesture.GestureFeature import *
from featureModules.objects.ObjectFeature import *
from featureModules.pose.PoseFeature import *
from featureModules.gaze.GazeFeature import *


# This took too many hours to find (https://github.com/conda/conda/issues/10897)
# add_dll_directory apparently doesn't work correctly without this
os.environ["CONDA_DLL_SEARCH_MODIFICATION_ENABLE"] = "1"

# tell the script where to find dll's for k4a, cuda, etc.
# body tracking sdk's tools should contain everything
os.add_dll_directory("C:\\Program Files\\Azure Kinect Body Tracking SDK\\tools")

import azure_kinect


if __name__ == "__main__":

    p = azure_kinect.Playback(r"C:\Users\brady\Desktop\Group_01-master.mkv")

    cameraMatrix, rotation, translation, distortion = p.get_calibration_matrices()


    shift = 7 # TODO what is this?
    gaze = GazeFeature(shift)
    gesture = GestureFeature(shift)
    objects = ObjectFeature()
    pose = PoseFeature()

    frames = 0
    while frames < 30:
        # load next frame
        color_image, depth_image, body_frame_info = p.get_frame()
        if color_image is None or depth_image is None:
            print("no color/depth image, skipping frame")
            continue

        framergb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        frame = cv2.cvtColor(color_image, cv2.IMREAD_COLOR)
        depth = depth_image

        w,h,c = color_image;
        bodies = body_frame_info["bodies"]
        deviceId = 0

        # run features
        blockStatus = {}
        blocks = objects.processFrame(color_image)
        pose.processFrame(bodies, frame)

        gaze.processFrame( bodies, w, h, rotation, translation, cameraMatrix, distortion, frame, framergb, depth, blocks, blockStatus)

        gesture.processFrame(deviceId, bodies, w, h, rotation, translation, cameraMatrix, distortion, frame, framergb, depth, blocks, blockStatus)

        cv.putText(frame, "FRAME:" + str(frames), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        cv.putText(frame, "DEVICE:" + str(int(deviceId)), (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

        cv.resize(frame, (640, 360))
        cv.imshow("OUTPUT", frame)
        cv.waitKey(1)

        frames += 1

    p.close()
