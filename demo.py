import os

import cv2 as cv

# from featureModules.gesture.GestureFeature import *
# from featureModules.objects.ObjectFeature import *
# from featureModules.pose.PoseFeature import *
# from featureModules.gaze.GazeFeature import *
from featureModules.asr.AsrFeature import *

import time
from threading import Event, Thread
import numpy as np
from ctypes import *

import sys

# tell the script where to find certain dll's for k4a, cuda, etc.
# body tracking sdk's tools should contain everything
os.add_dll_directory(r"C:\Program Files\Azure Kinect Body Tracking SDK\tools")
import azure_kinect

sys.path.append(r"C:\Program Files\Azure Kinect Body Tracking SDK\tools")


if __name__ == "__main__":

    shift = 7 # TODO what is this?
    # gaze = GazeFeature(shift)
    # gesture = GestureFeature(shift)
    # objects = ObjectFeature()
    # pose = PoseFeature()
    asr = AsrFeature()

    open = False
    attempts = 1
    while open == False and attempts <= 5:
        try:
            device = azure_kinect.Playback(rf"C:\Users\brady\Desktop\Group_01-master.mkv")
            # device = azure_kinect.Camera(0)
        except Exception as e:
            print(str(e))
            print("Error opening device trying again. Attempt " + str(attempts) + "\\5" + "\n")
            attempts+=1
            continue
        open = True
    
    if open == False:
        exit()
    
    device_id = 0
    cameraMatrix, rotation, translation, distortion = device.get_calibration_matrices()

    frame_count = 0
    while True:

        color_image, depth_image, body_frame_info = device.get_frame()
        if color_image is None or depth_image is None:
            print(f"DEVICE {device_id}: no color/depth image, skipping frame {frame_count}")
            frame_count += 1
            continue

        color_image = color_image[:,:,:3]
        depth = depth_image

        framergb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        frame = cv2.cvtColor(color_image, cv2.IMREAD_COLOR)

        h,w,_ = color_image.shape
        bodies = body_frame_info["bodies"]

        # run features
        blockStatus = {}
        blocks = []
        # blocks = objects.processFrame(framergb)
        # pose.processFrame(bodies, frame)
        # gaze.processFrame( bodies, w, h, rotation, translation, cameraMatrix, distortion, frame, framergb, depth, blocks, blockStatus)
        # gesture.processFrame(device_id, bodies, w, h, rotation, translation, cameraMatrix, distortion, frame, framergb, depth, blocks, blockStatus)
        asr.processFrame(device_id, bodies, w, h, rotation, translation, cameraMatrix, distortion, frame, framergb, depth, blocks, blockStatus)

        cv.putText(frame, "FRAME:" + str(frame_count), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        cv.putText(frame, "DEVICE:" + str(int(device_id)), (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

        frame = cv.resize(frame, (1280, 720))
        cv.imshow("output", frame)
        cv.waitKey(1)

        frame_count += 1

    device.close()
