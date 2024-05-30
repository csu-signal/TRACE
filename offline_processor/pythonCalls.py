import cv2
import mediapipe as mp
import os
import shutil
from featureModules.gesture.GestureFeature import *
from featureModules.objects.ObjectFeature import *
from featureModules.pose.PoseFeature import *
from featureModules.gaze.GazeFeature import *
from utils import *
import traceback
import numpy as np
import cv2
import os
from tkinter import *

#region initialize python

blockStatus = {}

shift = 7

keyFrame = {}
keyFrame[0] = np.zeros((360, 640, 3), dtype = "uint8")
keyFrame[1] = np.zeros((360, 640, 3), dtype = "uint8")
keyFrame[2] = np.zeros((360, 640, 3), dtype = "uint8")
keyFrame[3] = np.zeros((360, 640, 3), dtype = "uint8")

depthPaths = {}
depthPaths[0] = []
depthPaths[1] = []
depthPaths[2] = []

# feature modules
gaze = GazeFeature(shift)
gesture = GestureFeature(shift)
objects = ObjectFeature()
pose = PoseFeature()
#TODO add ASR

#region GUI setup

# create root window
root = Tk()

IncludePointing = IntVar(value=1)   
IncludeObjects = IntVar(value=1)   
IncludeGaze = IntVar(value=1)
IncludeASR = IntVar(value=0)
IncludePose = IntVar(value=1)
 
# root window title and dimension
root.title("Output Options")
root.geometry('350x200')
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

Button1.pack()
Button2.pack()
Button3.pack()
Button4.pack()
Button5.pack()

#endregion

#endregion

def concat_vh(list_2d): 
    return cv2.vconcat([cv2.hconcat(list_h)  
                        for list_h in list_2d]) 

def processFrameAzureBased(frame, depthPath, frameCount, deviceId, showOverlay, json, rotation, translation, cameraMatrix, dist):
    h, w, c = frame.shape
    depthPath = depthPath + str(int(frameCount)) + ".png"
    blockStatus = {}

    root.update()
    
    depthPaths[deviceId].append(depthPath)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.IMREAD_COLOR)

    try:
        depth = cv2.imread(depthPath, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        print(e)
        return

    if showOverlay:
        # Show the final output
        if(deviceId == 0):
            overlay = cv2.imread('5131.png')
        if(deviceId == 2):
            overlay = cv2.imread('5131sub1.png')
        if(deviceId == 1):
            overlay = cv2.imread('5131sub2.png')
        vis = cv2.addWeighted(overlay,0.5,frame,0.7,0)

        vis = cv2.resize(vis, (960, 640))
        cv2.imshow("OVERLAY " + str(deviceId), vis)
        cv2.waitKey(1)

    #region process features

    blocks = []
    bodies = json["bodies"]

    if(IncludeObjects.get() == 1):
        blocks = objects.processFrame(framergb)

    if(IncludePose.get() == 1):
        pose.processFrame(bodies, frame)

    if(IncludeGaze.get() == 1):
        gaze.processFrame( bodies, w, h, rotation, translation, cameraMatrix, dist, frame, framergb, depth, blocks, blockStatus)
       
    if(IncludePointing.get() == 1):
        gesture.processFrame(deviceId, bodies, w, h, rotation, translation, cameraMatrix, dist, frame, framergb, depth, blocks, blockStatus)

    #if(IncludeASR.get() == 1):

    #endregion 

    cv2.putText(frame, "FRAME:" + str(int(frameCount)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(frame, "DEVICE:" + str(int(deviceId)), (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    
    keyFrame[deviceId] = cv2.resize(frame, (640, 360))

    # remove depth data
    if(frameCount == 0 or frameCount % 100 == 0):
        for path in depthPaths[deviceId]:
            try: 
                os.remove(path)
            except Exception as e:
                 print(e)
        depthPaths[deviceId].clear()
        
#region cpp process call ins

# def callOpenFrameHardDrive(data, depthPath, frameCount, deviceId, showOverlay, frameJson, cameraJson):
#     try:
#         encoding = 'utf-8'
#         path = data.decode(encoding)   
#         frameData = frameJson.decode(encoding) 
#         calibration = cameraJson.decode(encoding)
#         pathDepth = depthPath.decode(encoding)
#         cameraMatrix, rotation, translation, dist = getCalibrationFromFile(json.loads(calibration))  
#         frame = cv2.imread(path)
    
#         processFrameAzureBased(frame, pathDepth, frameCount, deviceId, showOverlay, json.loads(frameData), rotation, translation, cameraMatrix, dist)
#         os.remove(path)
#         return 1
#     except Exception as e:
#         print(e) 
#         print(traceback.format_exc()) 

def openFrameBytes(bytes, depthPath, frameCount, deviceId, showOverlay, frameJson, cameraJson):
    try:
        encoding = 'utf-8'
        frameData = frameJson.decode(encoding) 
        calibration = cameraJson.decode(encoding)
        pathDepth = depthPath.decode(encoding)
        cameraMatrix, rotation, translation, dist = getCalibrationFromFile(json.loads(calibration))  
    
        return processFrameAzureBased(bytes, pathDepth, frameCount, deviceId, showOverlay, json.loads(frameData), rotation, translation, cameraMatrix, dist)
    except Exception as e:
        print(e) 
        print(traceback.format_exc())

def createFolder(data):
    try:
        encoding = 'utf-8'
        path = data.decode(encoding)   
        print(path)

        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
    except Exception as e:
        print(e) 

def showOutput():
    # concat = concat_vh([[keyFrame[1], keyFrame[0]], [keyFrame[2], keyFrame[3]]])
    # concat = concat_vh([[keyFrame[1], keyFrame[0]]])
    cv2.imshow("OUTPUT", keyFrame[0])
    cv2.waitKey(1)

#endregion
