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

from model import create_model
from config import (
    NUM_CLASSES, DEVICE, CLASSES
)

#TODO python dialog with check boxes to show different overlays

#region initialize python

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, static_image_mode= True, min_detection_confidence=0.6, min_tracking_confidence= 0)

loaded_model = joblib.load(".\\bestModel-pointing.pkl")
devicePoints = {}
keyFrame = {}
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
# load the best model and trained weights - for object detection
model = create_model(num_classes=NUM_CLASSES)
checkpoint = torch.load('.\\best_model-objects.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.to(DEVICE).eval()

# define the detection threshold...
# ... any detection having score below this will be discarded
detection_threshold = 0.6
RESIZE_TO = (512, 512)

#endregion
#endregion

#region point process utils

def findHands(frame, framergb, bodyId, handedness, box, points, cameraMatrix, dist, depth, blocks):   
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

                    landmarks = []
                    for lm in handslms.landmark:
                        lmx, lmy = int(lm.x * w) + box[0], int(lm.y * h) + box[1] 
                        #cv2.circle(frame, (lmx, lmy), radius=2, thickness= 2, color=(0,0,255))
                        landmarks.append([lmx, lmy])

                    if prediction[0][0] >= 0.2:
                        points.append(landmarks)
                        tx, ty, tip3D, bx, by, base3D, nextPoint, success = processPoint(handslms, box, w, h, cameraMatrix, dist, depth)
                        #cv2.putText(frame, str(success), (50,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)

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

                            for block in blocks:
                                targetPoint = [(block.p1[0]),(block.p1[1])]

                                try:
                                    object3D, success = convertTo3D(cameraMatrix, dist, depth, int(targetPoint[0]), int(targetPoint[1]))
                                    if(success == ParseResult.InvalidDepth):
                                        print("Ignoring invalid depth")
                                        continue
                                except:
                                    continue

                                block.target = cone.ContainsPoint(object3D[0], object3D[1], object3D[2], frame, True)
                                if(block.target):
                                    cv2.rectangle(frame, 
                                        (int(block.p1[0] * 2**shift), int(block.p1[1] * 2**shift)),
                                        (int(block.p2[0] * 2**shift), int(block.p2[1] * 2**shift)),
                                        color=(255,0,0),
                                        thickness=5, 
                                        shift=shift)
                                    cv2.circle(frame, (int(targetPoint[0] * 2**shift), int(targetPoint[1] * 2**shift)), radius=10, color=(0,0,0), thickness=10, shift=shift)
                                else:
                                    cv2.rectangle(frame, 
                                        (int(block.p1[0] * 2**shift), int(block.p1[1] * 2**shift)),
                                        (int(block.p2[0] * 2**shift), int(block.p2[1] * 2**shift)),
                                        color=(0,0,255),
                                        thickness=5, 
                                        shift=shift)
                                    cv2.circle(frame, (int(targetPoint[0] * 2**shift), int(targetPoint[1] * 2**shift)), radius=10, color=(0,0,0), thickness=10, shift=shift)  

def concat_vh(list_2d): 
    return cv2.vconcat([cv2.hconcat(list_h)  
                        for list_h in list_2d]) 

def processFrameAzureBased(frame, depthPath, frameCount, deviceId, showOverlay, json, rotation, translation, cameraMatrix, dist):
    points = []
    h, w, c = frame.shape
    depthPath = depthPath + str(int(frameCount)) + ".png"
    
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

    #region object detections
        
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
        outputs = model(image.to(DEVICE))
    
    # object rendering
    # load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]  
    blocks = []
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

    if(frameCount == 0 or frameCount % 500 == 0):
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
    concat = concat_vh([[keyFrame[1], keyFrame[0]], [keyFrame[2], keyFrame[3]]])
    cv2.imshow("OUTPUT", concat)
    cv2.waitKey(1)

#endregion
