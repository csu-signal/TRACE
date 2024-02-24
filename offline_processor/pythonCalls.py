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
        return processFrame(frame, frameCount, 0)
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
     

def openFramePath(path, frameCount):
    try:
        print(path)
        frame = cv2.imread(path)
        return processFrame(frame, frameCount, 0)
    except Exception as e:
        print(e) 

def openFrameBytes(bytes, frameCount, deviceId, frameJson, cameraJson):
    try:
        encoding = 'utf-8'
        frameData = frameJson.decode(encoding) 
        calibration = cameraJson.decode(encoding)

        #print(frameData)
        #print(calibration)

        cameraMatrix, rotation, translation, dist = getCalibrationFromFile(json.loads(calibration))  
        #print(cameraMatrix)
    
        #return processFrame(bytes, frameCount, deviceId)
        return processFrameAzureBased(bytes, frameCount, deviceId, json.loads(frameData), rotation, translation, cameraMatrix, dist)
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

def processFrameAzureBased(frame, frameCount, deviceId, json, rotation, translation, cameraMatrix, dist):
    #print(json)
    points = []
    h, w, c = frame.shape
    # Flip the frame horizontal
    # frame = cv2.flip(frame, 1)

    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.IMREAD_COLOR)

    bodies = json["bodies"]
    for bodyIndex, body in enumerate(bodies):  
        #print(body)
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
                # if(handedness == Handedness.Right):
                #     mpDraw.draw_landmarks(rightBox, hand, mpHands.HAND_CONNECTIONS)
                # else:
                    # mpDraw.draw_landmarks(frame, hand, mpHands.HAND_CONNECTIONS)
                for point in hand:
                    cv2.circle(frame, point, radius=2, thickness= 2, color=(0,255,0))

        frame = cv2.resize(frame, (960, 640))
        cv2.imshow("CAMERA " + str(int(deviceId)), frame)
        cv2.waitKey(1)

def processFrame(frame, frameCount, deviceId, json):
    points = []
    x , y, c = frame.shape

    # Flip the frame horizontal
    frame = cv2.flip(frame, 1)

    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
    
    # Get hand landmark prediction
    result = hands.process(framergb)
    result = hands.process(framergb)

    # post process the result
    if result.multi_hand_landmarks:
            #landmarks = []
            for handslms in result.multi_hand_landmarks:
                # for lm in handslms.landmark:
                #     # print(id, lm)
                #     lmx = int(lm.x * x)
                #     lmy = int(lm.y * y)

                #     landmarks.append([lmx, lmy])

                # Drawing landmarks on frames
                normalized = processHands(frame, handslms)
                prediction = loaded_model.predict([normalized])
                #mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
                for i in range(len(prediction)):
                    if prediction[i] == 0:
                        points.append(handslms)

    devicePoints[deviceId] = points
    if(deviceId == 0):
        cv2.putText(frame, "FRAME:" + str(int(frameCount)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        for key in devicePoints:
            if(len(devicePoints[key]) == 0):
                cv2.putText(frame, "NO POINTS, DEVICE:" + str(int(key)), (50,100 + (50 * int(key))), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "POINTS DETECTED, DEVICE:" + str(int(key)), (50,100 + (50 * int(key))), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            for hand in devicePoints[key]:
                mpDraw.draw_landmarks(frame, hand, mpHands.HAND_CONNECTIONS)

        frame = cv2.resize(frame, (960, 640))
        cv2.imshow("CAMERA " + str(int(deviceId)), frame)
        cv2.waitKey(1)

    # Show the final output
    # if(deviceId == 0):
    #     overlay = cv2.imread('5131.png')
    # if(deviceId == 2):
    #     overlay = cv2.imread('5131sub1.png')
    # if(deviceId == 1):
    #     overlay = cv2.imread('5131sub2.png')
    # vis = cv2.addWeighted(overlay,0.5,frame,0.7,0)

    #cv2.imwrite(".\\Camera1_output\\" + str(frameCount) + ".png", frame)

    # vis = cv2.resize(vis, (960, 640))
    # cv2.imshow("OVERLAY " + str(deviceId), vis)
    # cv2.waitKey(1)
                
    return 1 

# test = "{\"cx\":962.8074340820313,\"cy\":550.3942260742188,\"fx\":911.5870361328125,\"fy\":911.545166015625,\"k1\":0.42810821533203125,\"k2\":-2.7314584255218506,\"k3\":1.6649699211120605,\"k4\":0.3093422055244446,\"k5\":-2.5492746829986572,\"k6\":1.5847551822662354,\"p1\":0.0004034260637126863,\"p2\":-0.00010884667426580563,\"rotation\":[0.9999893307685852,0.0016191748436540365,-0.004320160020142794,-0.001163218286819756,0.9946256279945374,0.103530153632164,0.004464575555175543,-0.10352402180433273,0.9946169257164001],\"translation\":[-32.072898864746094,-2.0814547538757324,3.8745241165161133]}" 
# getCalibrationFromFile(json.loads(test))  

# for x in range(125):
#     print(x)
#     openFramePath("C:\\Users\\vanderh\\GitHub\\Camera_Calibration\\offline_processor\\build\\bin\\Debug\\Camera1_testData\\" + str(x) + ".png", x)  
