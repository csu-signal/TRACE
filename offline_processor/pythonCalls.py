import math
import cv2
import mediapipe as mp
import os
import shutil
import joblib
from utils import *
import numpy

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=4, min_detection_confidence=0.6)
mpDraw = mp.solutions.drawing_utils
loaded_model = joblib.load(".\\bestModel.pkl")
devicePoints = {}
shift = 7

def myabs(x):
    return math.fabs(x)

def openFrame(data, frameCount):
    try:
        encoding = 'utf-8'
        path = data.decode(encoding)   
        print(path)
        frame = cv2.imread(path)
        print("TODO setup azure based path rendering")
    except Exception as e:
        print(e) 

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

def openFrameBytes(bytes, frameCount, deviceId, showOverlay, frameJson, cameraJson):
    try:
        encoding = 'utf-8'
        frameData = frameJson.decode(encoding) 
        calibration = cameraJson.decode(encoding)
        cameraMatrix, rotation, translation, dist = getCalibrationFromFile(json.loads(calibration))  
    
        return processFrameAzureBased(bytes, frameCount, deviceId, showOverlay, json.loads(frameData), rotation, translation, cameraMatrix, dist)
    except Exception as e:
        print(e) 

def findHands(frame, framergb, bodyId, handedness, box, points):   
    dotColor = dotColors[bodyId % len(dotColors)]
    cv2.rectangle(frame, 
        (box[0]* 2**shift, box[1]* 2**shift), 
        (box[2]* 2**shift, box[3]* 2**shift), 
        color=dotColor,
        thickness=3, 
        shift=shift)
        
    with mpHands.Hands(max_num_hands=1, min_detection_confidence=0.6) as hands:
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
                        prediction = loaded_model.predict([normalized])
                        landmarks = []
                        for lm in handslms.landmark:
                            #print(lm)
                            lmx, lmy = int(lm.x * w) + box[0], int(lm.y * h) + box[1] 
                            cv2.circle(frame, (lmx, lmy), radius=2, thickness= 2, color=(0,0,255))
                            landmarks.append([lmx, lmy])
                        for i in range(len(prediction)):
                            if prediction[i] == 0:
                                print("Point Found")
                                points.append(landmarks)

def processFrameAzureBased(frame, frameCount, deviceId, showOverlay, json, rotation, translation, cameraMatrix, dist):
    points = []
    h, w, c = frame.shape

    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.IMREAD_COLOR)

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

    bodies = json["bodies"]
    for _, body in enumerate(bodies):  
        leftXAverage, leftYAverage, rightXAverage, rightYAverage = getAverageHandLocations(body, w, h, rotation, translation, cameraMatrix, dist)
        rightBox = createBoundingBox(rightXAverage, rightYAverage)
        leftBox = createBoundingBox(leftXAverage, leftYAverage)
        findHands(frame,
                framergb,
                int(body['body_id']),
                Handedness.Left, 
                leftBox,
                points) 
        findHands(frame,
                framergb,
                int(body['body_id']),
                Handedness.Right, 
                rightBox,
                points)
    devicePoints[deviceId] = points

    if(deviceId == 0):
        cv2.putText(frame, "FRAME:" + str(int(frameCount)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        for key in devicePoints:
            if(len(devicePoints[key]) == 0):
                cv2.putText(frame, "NO POINTS, DEVICE:" + str(int(key)), (50,100 + (50 * int(key))), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "POINTS DETECTED, DEVICE:" + str(int(key)), (50,100 + (50 * int(key))), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            for hand in devicePoints[key]:
                for point in hand:
                    cv2.circle(frame, point, radius=2, thickness= 2, color=(0,255,0))

        frame = cv2.resize(frame, (960, 640))
        cv2.imshow("CAMERA " + str(int(deviceId)), frame)
        cv2.waitKey(1)
