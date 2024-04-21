import math
import cv2
import mediapipe as mp
import os
import shutil
import joblib
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

from model import create_model
from config import (
    NUM_CLASSES, DEVICE, CLASSES
)

#TODO python dialog with check boxes to show different overlays

#region gaze util methods

def euclideanLoss(y_true, y_pred):
    return K.mean(K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1)))

def predict_gaze(model, image, faces, heads):
    preds = model.predict([np.array(image),np.array(faces),np.array(heads)])
    return preds

#endregion

#region initialize python

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, static_image_mode= True, min_detection_confidence=0.4, min_tracking_confidence= 0)
gazeCount = []
gazeCount.append(0)

loaded_model = joblib.load(".\\bestModel-pointing.pkl")
devicePoints = {}
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

#region initalize object detections 

print("Torch Device " + str(DEVICE))
print("Python version " + str(platform.python_version()))
# load the best objectModel and trained weights - for object detection
objectModel = create_model(num_classes=NUM_CLASSES)
checkpoint = torch.load('.\\best_model-objects.pth', map_location=DEVICE)
objectModel.load_state_dict(checkpoint['model_state_dict'], strict=False)
objectModel.to(DEVICE).eval()

# define the detection threshold...
# ... any detection having score below this will be discarded
detection_threshold = 0.8
RESIZE_TO = (512, 512)

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
IncludeASR = IntVar(value=1)
 
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

Button1.pack()
Button2.pack()
Button3.pack()
Button4.pack()

#endregion

#region ASR socket 

try: 
   connected = True
   print("Attempting socket connection")
   s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
   s.settimeout(0.1)
   s.connect(('10.84.208.54', 9999)) #ipv4 address of rosch
except socket.error:
    connected = False
    print("failed to connect to ASR socket")

#endregion
#endregion

#region point process utils

def findHands(frame, framergb, bodyId, handedness, box, points, cameraMatrix, dist, depth, blocks):   
    ## to help visualize where the hand localization is focused
    # dotColor = dotColors[bodyId % len(dotColors)]
    # cv2.rectangle(frame, 
    #     (box[0]* 2**shift, box[1]* 2**shift), 
    #     (box[2]* 2**shift, box[3]* 2**shift), 
    #     color=dotColor,
    #     thickness=3, 
    #     shift=shift)

    #media pipe on the sub box only
    frameBox = framergb[box[1]:box[3], box[0]:box[2]]
    h, w, c = frameBox.shape

    if(h > 0 and w > 0):
        results = hands.process(frameBox)

        handedness_result = results.multi_handedness
        if results.multi_hand_landmarks:
            for index, handslms in enumerate(results.multi_hand_landmarks):
                returned_handedness = handedness_result[index].classification[0]
                if(returned_handedness.label == handedness.value):
                    normalized = processHands(frame, handslms)
                    prediction = loaded_model.predict_proba([normalized])
                    #print(prediction)

                    if prediction[0][0] >= 0.3:
                        landmarks = []
                        for lm in handslms.landmark:
                            lmx, lmy = int(lm.x * w) + box[0], int(lm.y * h) + box[1] 
                            #cv2.circle(frame, (lmx, lmy), radius=2, thickness= 2, color=(0,0,255))
                            landmarks.append([lmx, lmy])

                        points.append(landmarks)
                        tx, ty, mediaPipe8, bx, by, mediaPipe5, nextPoint, success = processPoint(landmarks, box, w, h, cameraMatrix, dist, depth)
                        #cv2.putText(frame, str(success), (50,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)

                        if success == ParseResult.Success:
                            point8_2D = convert2D(mediaPipe8, cameraMatrix, dist)
                            point5_2D = convert2D(mediaPipe5, cameraMatrix, dist) 
                            pointExtended_2D = convert2D(nextPoint, cameraMatrix, dist)

                            point_8 = (int(point8_2D[0] * 2**shift),int(point8_2D[1] * 2**shift))
                            point_5 = (int(point5_2D[0] * 2**shift),int(point5_2D[1] * 2**shift))
                            point_Extended = (int(pointExtended_2D[0] * 2**shift),int(pointExtended_2D[1] * 2**shift))

                            cv2.line(frame, point_8, point_Extended, color=(0, 165, 255), thickness=5, shift=shift)
                            cv2.line(frame, point_5, point_8, color=(0, 165, 255), thickness=5, shift=shift)

                            cv2.circle(frame, point_8, radius=15, color=(255,0,0), thickness=15, shift=shift)
                            cv2.circle(frame, (int(tx), int(ty)), radius=4, color=(0, 0, 255), thickness=-1)
                            cv2.circle(frame, point_5, radius=15, color=(255,0,0), thickness=15, shift=shift)
                            cv2.circle(frame, (int(bx), int(by)), radius=4, color=(0, 0, 255), thickness=-1)
                            cv2.circle(frame, point_Extended, radius=15, color=(255,0,0), thickness=15, shift=shift)

                            cone = ConeShape(mediaPipe5, nextPoint, 80, 100, cameraMatrix, dist)
                            cone.projectRadiusLines(shift, frame, True, False, False)
                            checkBlocks(blocks, blockStatus, cameraMatrix, dist, depth, cone, frame, shift, False)

def concat_vh(list_2d): 
    return cv2.vconcat([cv2.hconcat(list_h)  
                        for list_h in list_2d]) 

def processFrameAzureBased(frame, depthPath, frameCount, deviceId, showOverlay, json, rotation, translation, cameraMatrix, dist):
    points = []
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

    if(connected == True and IncludeASR.get() == 1):
        try:
            msg = s.recv(4096)
        except socket.error as e:
            err = e.args[0]
            if err != errno.EAGAIN and err != errno.EWOULDBLOCK:
                print(e)
        else:
            encoding = 'utf-8'
            output = msg.decode(encoding)  
            print(output)

            wrapped_text = textwrap.wrap(output, width=50)
            asrFrame = np.zeros((1080, 1920, 3), dtype = "uint8")

            for i, line in enumerate(wrapped_text):
                cv2.putText(asrFrame, str(line), (75,75 * (1+i)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)

            keyFrame[1] = cv2.resize(asrFrame, (640, 360))

    #region object detections
    blocks = []
    if(IncludeObjects.get() == 1):
        image = cv2.resize(framergb, RESIZE_TO)
        image = framergb.astype(np.float32)
        # make the pixel range between 0 and 1
        image /= 255.0
        # bring color channels to front
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        # convert to tensor
        image = torch.tensor(image, dtype=torch.float).cuda()
        # add batch dimension
        image = torch.unsqueeze(image, 0)
        with torch.no_grad():
            # get predictions for the current frame
            outputs = objectModel(image.to(DEVICE))
        
        # object rendering
        # load all detection to CPU for further operations
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]  
        found = []
        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            draw_boxes = boxes.copy()
            # get all the predicited class names
            pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

            # draw the bounding boxes and write the class name on top of it
            for j, box in enumerate(draw_boxes):
                class_name = pred_classes[j]
                if(found.__contains__(class_name)):
                    continue

                found.append(class_name)
                p1 = [box[0], box[1]]
                p2 = [box[2], box[3]]
                
                block = Block(float(class_name), p1, p2)
                blocks.append(block)
                
                print("Found Block: " + str(block.description))
                # print(str(p1))
                # print(str(p2))

                cv2.rectangle(frame, 
                    (int(p1[0] * 2**shift), int(p1[1] * 2**shift)),
                    (int(p2[0] * 2**shift), int(p2[1] * 2**shift)),
                    color=(255,255,255),
                    thickness=3, 
                    shift=shift)

    #endregion

    bodies = json["bodies"]

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
                    print(lenAB)

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
                        cv2.line(frame, (head_p1, head_p2), (pred_p1, pred_p2), thickness=5, shift=shift, color=(107,190,255))

                        cone = ConeShape(head3D, pred3D, 80, 100, cameraMatrix, dist)
                        cone.projectRadiusLines(shift, frame, True, False, True)
                        
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
        for _, body in enumerate(bodies):  
            leftXAverage, leftYAverage, rightXAverage, rightYAverage = getAverageHandLocations(body, w, h, rotation, translation, cameraMatrix, dist)
            rightBox = createBoundingBox(rightXAverage, rightYAverage)
            leftBox = createBoundingBox(leftXAverage, leftYAverage)
            findHands(frame,
                    framergb,
                    int(body['body_id']),
                    Handedness.Left, 
                    leftBox,
                    points,
                    cameraMatrix,
                    dist,
                    depth,
                    blocks) 
            findHands(frame,
                    framergb,
                    int(body['body_id']),
                    Handedness.Right, 
                    rightBox,
                    points,
                    cameraMatrix,
                    dist,
                    depth,
                    blocks)
        devicePoints[deviceId] = points

        for key in devicePoints:
            if(key == deviceId):
                if(len(devicePoints[key]) == 0):
                    cv2.putText(frame, "NO POINTS", (50,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "POINTS DETECTED", (50,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                for hand in devicePoints[key]:
                    for point in hand:
                        cv2.circle(frame, point, radius=2, thickness= 2, color=(0,255,0))

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

#endregion
        
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
    concat = concat_vh([[keyFrame[1], keyFrame[0]]])
    cv2.imshow("OUTPUT", concat)
    cv2.waitKey(1)

#endregion
