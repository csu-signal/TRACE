import math
import cv2
import mediapipe as mp
import os
import shutil
from featureModules.gesture.GestureFeature import *
from featureModules.objects.ObjectFeature import *
from utils import *
import traceback
import numpy as np
import cv2
import os
import torch
import platform
from tensorflow import keras
from tensorflow.keras.metrics import categorical_accuracy
from Face_Detection import load_frame, load_frame_azure
from mtcnn import MTCNN
from tkinter import *
import socket
import errno
from time import sleep
import textwrap 
import math
import torch.nn as nn
import select

#region gaze util methods

def euclideanLoss(y_true, y_pred):
    return K.mean(K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1)))

def predict_gaze(model, image, faces, heads):
    preds = model.predict([np.array(image),np.array(faces),np.array(heads)])
    return preds

class SkeletonPoseClassifier(nn.Module):
    """
    Base model, input single body, binary output. Two feedforward layers.
    Note: label of a frame is a very strong predictor of the next, how to incorporate without risk?
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(SkeletonPoseClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

#endregion

#region initialize python

gazeCount = []
gazeCount.append(0)

keyFrame = {}
gazeHead = {}
gazePred = {}

gazeHeadAverage = {}
gazePredAverage = {}

blockStatus = {}

shift = 7

keyFrame[0] = np.zeros((360, 640, 3), dtype = "uint8")
keyFrame[1] = np.zeros((360, 640, 3), dtype = "uint8")
keyFrame[2] = np.zeros((360, 640, 3), dtype = "uint8")
keyFrame[3] = np.zeros((360, 640, 3), dtype = "uint8")

depthPaths = {}
depthPaths[0] = []
depthPaths[1] = []
depthPaths[2] = []

# feature modules
gesture = GestureFeature(shift)
objects = ObjectFeature()

#region initalize pose models

#  required arguments
input_size = 224
hidden_size = 300
output_size = 1
  
# initialize a model 
leftModel = SkeletonPoseClassifier(input_size = input_size,hidden_size=hidden_size,output_size=output_size)
leftModel.load_state_dict(torch.load(".\\skeleton_pose_classifier_left.pt"))
leftModel.eval()

middleModel = SkeletonPoseClassifier(input_size = input_size,hidden_size=hidden_size,output_size=output_size)
middleModel.load_state_dict(torch.load(".\\skeleton_pose_classifier_middle.pt"))
middleModel.eval()

rightModel = SkeletonPoseClassifier(input_size = input_size,hidden_size=hidden_size,output_size=output_size)
rightModel.load_state_dict(torch.load(".\\skeleton_pose_classifier_right.pt"))
rightModel.eval()

#endregion

#region initalize gaze detections

faceDetector = MTCNN()
gazeModel = keras.models.load_model(".\\Model\\1", custom_objects={'euclideanLoss': euclideanLoss,
                                                                 'categorical_accuracy': categorical_accuracy})
#endregion

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

#region ASR socket 

# try: 
# connected = []
# connected.append(False)

# last_data = []
# last_data.append(None)

# client = []
# client.append(None)

# #next create a socket object 
# server_ip='192.168.0.113'
# server_port=9999
# s = socket.socket()         
# print ("Server Socket successfully created")

# s.bind((server_ip, server_port))         
# print ("socket binded to %s" %(server_port))
# print ("Server IP address:", server_ip)

# # put the socket into listening mode 
# s.listen(2)     
# print ("socket is listening")    

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

    # if(IncludeASR.get() == 1):
    #     try:
    #         if(connected[0] == False):
    #             client[0], addr = s.accept()  
    #             print ('Got connection from', addr )
    #             connected[0] = True
    #             client[0].send('Thank you for connecting'.encode())
    #             client[0].setblocking(0)
    #         else:
    #             ready = select.select([client[0]], [], [], 0.1)
    #             if ready[0]:
    #                 data = client[0].recv(1024).decode()

    #                 # Only print the data if it's new
    #                 if data != last_data[0]:
    #                     last_data[0] = data  # Update the last received data 
    #                     print(data)
                          
    #                     # wrapped_text = textwrap.wrap(data, width=50)
    #                     # asrFrame = np.zeros((1080, 1920, 3), dtype = "uint8")

    #                     # for i, line in enumerate(wrapped_text):
    #                     #     cv2.putText(asrFrame, str(line), (75,75 * (1+i)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)

    #                     # keyFrame[1] = cv2.resize(asrFrame, (640, 360))
    #     except socket.error as e:
    #         print(e.args[0])

    #region process features

    blocks = []
    if(IncludeObjects.get() == 1):
        blocks = objects.processFrame(framergb)

    bodies = json["bodies"]

    #region pose detection
    if(IncludePose.get() == 1):
        left_position = -400
        middle_position = 400

        # left_position = 800
        # middle_position = 1200

        # cv2.circle(frame, (int(left_position), 800), radius=15, color=(255,0,0), thickness=15)
        # cv2.circle(frame, (int(middle_position), 800), radius=15, color=(255,0,0), thickness=15)

        for b in bodies:
            # points2D, _ = cv2.projectPoints(
            #         np.array(b['joint_positions'][1]), 
            #         rotation,
            #         translation,
            #         cameraMatrix,
            #         dist)  
            #x = points2D[0][0][0]
            x = b['joint_positions'][1][0]
            #print(x)

            if x < left_position:
               # print("left")
                poseModel = leftModel
                body = b
                position = "left"
            elif x > left_position and x < middle_position:
                #print("middle")
                poseModel = middleModel
                body = b
                position = "middle"
            else:
                #print("right")
                poseModel = rightModel
                body = b
                position = "right"

            # print(b['joint_positions'][1][0])
            # if b['joint_positions'][1][0] > left_position:
            #     print("left")
            #     poseModel = leftModel
            #     body = b
            #     position = "left"
            # elif b['joint_positions'][1][0] < middle_position:
            #     print("middle")
            #     poseModel = middleModel
            #     body = b
            #     position = "middle"
            # else:
            #     print("right")
            #     poseModel = rightModel
            #     body = b
            #     position = "right"

            tensors = []
            orientation_data = body['joint_orientations']
            position_data = body['joint_positions']
            o = torch.tensor(orientation_data).flatten()
            p = torch.tensor(position_data).flatten() / 1000 # normalize to scale of orientations
            tensors.append(torch.concat([o, p])) # concatenating orientation to position

            output = poseModel(torch.stack(tensors))
            # prediction = int(torch.argmax(output))
            prediction = output.detach().numpy()[0][0] > 0.5
            
            # print("Prediction: " + str(prediction))
            # print("Output: " + str(output))

            engagement = "leaning out" if prediction == 0 else "leaning in"
            color = (255,0,0) if prediction == 0 else (39,142,37)
            if position == "left":
                cv2.putText(frame, "P1: " + engagement, (50,200), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            elif position == "middle":
                cv2.putText(frame, "P2: " + engagement, (50,250), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "P3: " + engagement, (50,300), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    #endregion

    #region gaze detections

    if(IncludeGaze.get() == 1):
        #faces,heads,images=load_frame(frame,framergb,faceDetector,shift)
        faces,heads,images,bodyIds=load_frame_azure(frame,framergb,bodies, rotation, translation, cameraMatrix, dist, shift)
        if(len(faces) > 0):
            preds = predict_gaze(gazeModel, images, faces, heads)
            gazeCount[0] += 1 
            for index, head in enumerate(heads):
                key = bodyIds[index]
                #print(key)
                if key not in gazeHead:
                    gazeHead[key] = []
                    gazePred[key] = []
                    
                gazeHead[key].append([(heads[index][0] * w), (heads[index][1] * h)])
                gazePred[key].append([(preds[0][index][0] * w), (preds[0][index][1] * h)])
        
                if(len(gazePred[key]) == 5):
                    sumx = 0
                    sumy = 0
                    for point in gazeHead[key]: 
                        sumx += point[0]
                        sumy += point[1]

                    headX_average = int(sumx / 5)
                    headY_average = int(sumy / 5)

                    sumx = 0
                    sumy = 0
                    for point in gazePred[key]: 
                        sumx += point[0]
                        sumy += point[1]

                    predX_average = int(sumx / 5)
                    predY_average = int(sumy / 5)

                    lenAB = math.sqrt(pow(headX_average - predX_average, 2.0) + pow(headY_average - predY_average, 2.0))
                    #print(lenAB)

                    length = 500
                    if(lenAB < length):
                        # print("Made it into the length update")
                        # print("Before")
                        # print(predX_average)
                        # print(predY_average)
                        unitSlopeX = (predX_average-headX_average) / lenAB
                        unitSlopeY = (predY_average-headY_average) / lenAB

                        predX_average = int(headX_average + (unitSlopeX * length))
                        predY_average = int(headY_average + (unitSlopeY * length))
                        # print("After")
                        # print(predX_average)
                        # print(predY_average)

                        # predX_average = int(predX_average + (predX_average - headX_average) / lenAB * (length - lenAB))
                        # predY_average = int(predY_average + (predY_average - predX_average) / lenAB * (length - lenAB))

                    head3D, h_Success = convertTo3D(cameraMatrix, dist, depth, headX_average, headY_average)
                    pred3D, p_Success = convertTo3D(cameraMatrix, dist, depth, predX_average, predY_average)

                    if(h_Success == ParseResult.Success and p_Success == ParseResult.Success):
                        pred_p1 = int((predX_average) * 2**shift)
                        pred_p2 = int((predY_average) * 2**shift)

                        head_p1 = int((headX_average) * 2**shift)
                        head_p2 = int((headY_average) * 2**shift)
                        cv2.line(frame, (head_p1, head_p2), (pred_p1, pred_p2), thickness=5, shift=shift, color=(255, 107, 170))

                        cone = ConeShape(head3D, pred3D, 80, 100, cameraMatrix, dist)
                        cone.projectRadiusLines(shift, frame, False, False, True)
                        
                        checkBlocks(blocks, blockStatus, cameraMatrix, dist, depth, cone, frame, shift, True)
                    
                    # for key in gazeHead:
                    #     print(key)
                    copyHead = gazeHead[key]
                    #print(copyHead)

                    gazeHead[key] = []
                    gazeHead[key].append(copyHead[1]) 
                    gazeHead[key].append(copyHead[2])
                    gazeHead[key].append(copyHead[3])
                    gazeHead[key].append(copyHead[4])

                    copyPred = gazePred[key]
                    gazePred[key] = []
                    gazePred[key].append(copyPred[1]) 
                    gazePred[key].append(copyPred[2])
                    gazePred[key].append(copyPred[3])
                    gazePred[key].append(copyPred[4])

    #endregion 

    if(IncludePointing.get() == 1):
        gesture.processFrame(deviceId, bodies, w, h, rotation, translation, cameraMatrix, dist, frame, framergb, depth, blocks, blockStatus)

    #endregion 

    cv2.putText(frame, "FRAME:" + str(int(frameCount)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(frame, "DEVICE:" + str(int(deviceId)), (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    
    keyFrame[deviceId] = cv2.resize(frame, (640, 360))

    if(frameCount == 0 or frameCount % 100 == 0):
        for path in depthPaths[deviceId]:
            try: 
                os.remove(path)
            except Exception as e:
                 print(e)
        depthPaths[deviceId].clear()
        
#region cpp process call ins (for point detection)    

def callOpenFrameHardDrive(data, depthPath, frameCount, deviceId, showOverlay, frameJson, cameraJson):
    try:
        encoding = 'utf-8'
        path = data.decode(encoding)   
        frameData = frameJson.decode(encoding) 
        calibration = cameraJson.decode(encoding)
        pathDepth = depthPath.decode(encoding)
        cameraMatrix, rotation, translation, dist = getCalibrationFromFile(json.loads(calibration))  
        frame = cv2.imread(path)
    
        processFrameAzureBased(frame, pathDepth, frameCount, deviceId, showOverlay, json.loads(frameData), rotation, translation, cameraMatrix, dist)
        os.remove(path)
        return 1
    except Exception as e:
        print(e) 
        print(traceback.format_exc()) 

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
