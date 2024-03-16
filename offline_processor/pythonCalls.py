import math
import cv2
import mediapipe as mp
import os
import shutil
import joblib
from utils import *
import numpy
import traceback

#TODO python dialog with check boxes to show different overlays

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=4, min_detection_confidence=0.6)
mpDraw = mp.solutions.drawing_utils
loaded_model = joblib.load(".\\bestModel.pkl")
devicePoints = {}
keyFrame = {}
shift = 7

keyFrame[0] = np.zeros((360, 640, 3), dtype = "uint8")
keyFrame[1] = np.zeros((360, 640, 3), dtype = "uint8")
keyFrame[2] = np.zeros((360, 640, 3), dtype = "uint8")
keyFrame[3] = np.zeros((360, 640, 3), dtype = "uint8")

def myabs(x):
    return math.fabs(x)

def openFrame(data, depthPath, frameCount, deviceId, showOverlay, frameJson, cameraJson):
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

def findHands(frame, framergb, bodyId, handedness, box, points, cameraMatrix, dist, depth):   
    # dotColor = dotColors[bodyId % len(dotColors)]
    # cv2.rectangle(frame, 
    #     (box[0]* 2**shift, box[1]* 2**shift), 
    #     (box[2]* 2**shift, box[3]* 2**shift), 
    #     color=dotColor,
    #     thickness=3, 
    #     shift=shift)
        
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
                        prediction = loaded_model.predict_proba([normalized])
                        landmarks = []
                        for lm in handslms.landmark:
                            lmx, lmy = int(lm.x * w) + box[0], int(lm.y * h) + box[1] 
                            #cv2.circle(frame, (lmx, lmy), radius=2, thickness= 2, color=(0,0,255))
                            landmarks.append([lmx, lmy])

                        print(prediction)
                        if prediction[0][0] >= 0.2:
                            print("Point Found")
                            points.append(landmarks)
                            tx, ty, tip3D, bx, by, base3D, nextPoint, success = processPoint(handslms, box, w, h, cameraMatrix, dist, depth)
                            cv2.putText(frame, str(success), (50,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)

                            if success == ParseResult.Success:
                                pointTip2D = convert2D(tip3D, cameraMatrix, dist)
                                pointBase2D = convert2D(base3D, cameraMatrix, dist) 
                                pointNext2D = convert2D(nextPoint, cameraMatrix, dist)

                                pointTip = (int(pointTip2D[0] * 2**shift),int(pointTip2D[1] * 2**shift))
                                pointBase = (int(pointBase2D[0] * 2**shift),int(pointBase2D[1] * 2**shift))
                                pointNext = (int(pointNext2D[0] * 2**shift),int(pointNext2D[1] * 2**shift))

                                cv2.line(frame, pointTip, pointNext, color=(0, 165, 255), thickness=5, shift=shift)
                                cv2.line(frame, pointBase, pointTip, color=(0, 165, 255), thickness=5, shift=shift)

                                cv2.circle(frame, pointTip, radius=15, color=(255,0,0), thickness=15, shift=shift)
                                cv2.circle(frame, (int(tx), int(ty)), radius=4, color=(0, 0, 255), thickness=-1)
                                cv2.circle(frame, pointBase, radius=15, color=(255,0,0), thickness=15, shift=shift)
                                cv2.circle(frame, (int(bx), int(by)), radius=4, color=(0, 0, 255), thickness=-1)
                                cv2.circle(frame, pointNext, radius=15, color=(255,0,0), thickness=15, shift=shift)

                                cone = ConeShape(base3D, nextPoint, 25, 75, cameraMatrix, dist)
                                cone.projectRadiusLines(shift, frame, True, False)

def concat_vh(list_2d): 
    return cv2.vconcat([cv2.hconcat(list_h)  
                        for list_h in list_2d]) 

def processFrameAzureBased(frame, depthPath, frameCount, deviceId, showOverlay, json, rotation, translation, cameraMatrix, dist):
    points = []
    h, w, c = frame.shape
    depthPath = depthPath + str(int(frameCount)) + ".png"

    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.IMREAD_COLOR)

    try:
        depth = cv2.imread(depthPath, cv2.IMREAD_UNCHANGED)
    except Exception as e:
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

    bodies = json["bodies"]
    for bodyIndex, body in enumerate(bodies):  
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
                depth) 
        findHands(frame,
                framergb,
                int(body['body_id']),
                Handedness.Right, 
                rightBox,
                points,
                cameraMatrix,
                dist,
                depth)
    devicePoints[deviceId] = points

    cv2.putText(frame, "FRAME:" + str(int(frameCount)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(frame, "DEVICE:" + str(int(deviceId)), (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    
    for key in devicePoints:
        if(key == deviceId):
            if(len(devicePoints[key]) == 0):
                cv2.putText(frame, "NO POINTS", (50,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "POINTS DETECTED", (50,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            for hand in devicePoints[key]:
                for point in hand:
                    cv2.circle(frame, point, radius=2, thickness= 2, color=(0,255,0))

    keyFrame[deviceId] = cv2.resize(frame, (640, 360))
    cv2.waitKey(1)
    
    if(deviceId == 0):
        concat = concat_vh([[keyFrame[1], keyFrame[0]], [keyFrame[2], keyFrame[3]]])
        cv2.imshow("OUTPUT", concat)
        cv2.waitKey(1)

    try: 
        os.remove(depthPath)
    except: pass
