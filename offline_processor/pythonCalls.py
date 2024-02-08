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
        print("Made it to create folder")
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

def openFrameBytes(bytes, frameCount, deviceId):
    try:
        #print(bytes)
        #frame = cv2.imdecode(bytes, cv2.IMREAD_COLOR)
        return processFrame(bytes, frameCount, deviceId)
    except Exception as e:
        print(e)   

def processFrame(frame, frameCount, deviceId):
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
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    # print(id, lm)
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)

                    landmarks.append([lmx, lmy])

                # Drawing landmarks on frames
                normalized = processHands(frame, handslms)
                prediction = loaded_model.predict([normalized])
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
                for i in range(len(prediction)):
                    if prediction[i] == 0:
                        cv2.putText(frame, "POINT, HOLD", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(frame, "POINT, NO HOLD", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    else:
         cv2.putText(frame, "NO HANDS DETECTED", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)

    # Show the final output
    if(deviceId == 0):
        overlay = cv2.imread('5131.png')
    if(deviceId == 2):
        overlay = cv2.imread('5131sub1.png')
    if(deviceId == 1):
        overlay = cv2.imread('5131sub2.png')
    vis = cv2.addWeighted(overlay,0.5,frame,0.7,0)

    #cv2.imwrite(".\\Camera1_output\\" + str(frameCount) + ".png", frame)

    vis = cv2.resize(vis, (960, 640))
    cv2.imshow("OVERLAY " + str(deviceId), vis)

    frame = cv2.resize(frame, (960, 640))
    cv2.imshow("CAMERA " + str(deviceId), frame)

    cv2.waitKey(1)
    return 1   

# for x in range(125):
#     print(x)
#     openFramePath("C:\\Users\\vanderh\\GitHub\\Camera_Calibration\\offline_processor\\build\\bin\\Debug\\Camera1_testData\\" + str(x) + ".png", x)  
