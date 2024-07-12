import os
import cv2 as cv

from featureModules.gesture.GestureFeature import *
from featureModules.objects.ObjectFeature import *
from featureModules.pose.PoseFeature import *
from featureModules.gaze.GazeFeature import *
from featureModules.asr.AsrFeature import *
from featureModules.prop.PropExtractFeature import *
from featureModules.move.MoveFeature import *
from tkinter import *


# tell the script where to find certain dll's for k4a, cuda, etc.
# body tracking sdk's tools should contain everything
os.add_dll_directory(r"C:\Program Files\Azure Kinect Body Tracking SDK\tools")
import azure_kinect


if __name__ == "__main__":

    #region GUI setup
    root = Tk()

    IncludePointing = IntVar(value=1)   
    IncludeObjects = IntVar(value=1)   
    IncludeGaze = IntVar(value=1)
    IncludeASR = IntVar(value=1)
    IncludePose = IntVar(value=1)
    IncludeProp = IntVar(value=1)
    IncludeMove = IntVar(value=1)
    
    # root window title and dimension
    root.title("Output Options")
    # root.geometry('350x200')
    Button1 = Checkbutton(root, text = "Pointing",  
                        variable = IncludePointing, 
                        onvalue = 1, 
                        offvalue = 0, 
                        height = 2, 
                        width = 10) 

    Button2 = Checkbutton(root, text = "Objects", 
                        variable = IncludeObjects, 
                        onvalue = 1, 
                        offvalue = 0, 
                        height = 2, 
                        width = 10) 

    Button3 = Checkbutton(root, text = "Gaze", 
                        variable = IncludeGaze, 
                        onvalue = 1, 
                        offvalue = 0, 
                        height = 2, 
                        width = 10) 

    Button4 = Checkbutton(root, text = "ASR", 
                        variable = IncludeASR, 
                        onvalue = 1, 
                        offvalue = 0, 
                        height = 2, 
                        width = 10) 

    Button5 = Checkbutton(root, text = "Pose", 
                        variable = IncludePose, 
                        onvalue = 1, 
                        offvalue = 0, 
                        height = 2, 
                        width = 10) 

    Button6 = Checkbutton(root, text = "PropExtract", 
                        variable = IncludeProp, 
                        onvalue = 1, 
                        offvalue = 0, 
                        height = 2, 
                        width = 10) 

    Button7 = Checkbutton(root, text = "Move Classifier", 
                        variable = IncludeMove, 
                        onvalue = 1, 
                        offvalue = 0, 
                        height = 2, 
                        width = 10) 

    Button1.pack()
    Button2.pack()
    Button3.pack()
    Button4.pack()
    Button5.pack()
    Button6.pack()
    Button7.pack()

    #endregion

    shift = 7 # TODO what is this?
    gaze = GazeFeature(shift)
    gesture = GestureFeature(shift)
    objects = ObjectFeature()
    pose = PoseFeature()
    asr = AsrFeature([('Participant 1',2),('Participant 2',6),('Participant 3',15)], n_processors=3)
    # asr = AsrFeature([('Participant 1',1)], n_processors=1)
    prop = PropExtractFeature()
    move = MoveFeature()

    device = None
    attempts = 0
    while device is None and attempts < 5:
        try:
            # device = azure_kinect.Playback(rf"C:\Users\brady\Desktop\Group_01-master.mkv")
            device = azure_kinect.Camera(0)

        except Exception as e:
            attempts += 1
            print(str(e))
            print("Error opening device trying again. Attempt " + str(attempts) + "\\5" + "\n")
            continue
    
    if device is None:
        exit()
    
    device_id = 0
    cameraMatrix, rotation, translation, distortion = device.get_calibration_matrices()

    frame_count = 0
    while True:
        root.update()
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

        if(IncludeObjects.get() == 1):
            blocks = objects.processFrame(framergb)

        if(IncludePose.get() == 1):
            pose.processFrame(bodies, frame)

        if(IncludeGaze.get() == 1):
            gaze.processFrame( bodies, w, h, rotation, translation, cameraMatrix, distortion, frame, framergb, depth, blocks, blockStatus)
        
        if(IncludePointing.get() == 1):
             gesture.processFrame(device_id, bodies, w, h, rotation, translation, cameraMatrix, distortion, frame, framergb, depth, blocks, blockStatus)

        utterances = []
        if(IncludeASR.get() == 1):
            utterances = asr.processFrame(frame)
            if(IncludePointing.get() == 1):
                utterances = gesture.updateDemonstratives(utterances)

        utterances_and_props = []
        if(IncludeProp.get() == 1):
            utterances_and_props = prop.processFrame(frame, utterances)

        if(IncludeMove.get() == 1):
            move.processFrame(utterances_and_props, frame)

        cv.putText(frame, "FRAME:" + str(frame_count), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        cv.putText(frame, "DEVICE:" + str(int(device_id)), (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

        frame = cv.resize(frame, (1280, 720))
        cv.imshow("output", frame)
        cv.waitKey(1)

        frame_count += 1

    device.close()
