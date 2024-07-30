import argparse
import cv2
import numpy as np
import json
from enum import Enum
import mediapipe as mp
import demo.featureModules.utils as utils

class Joint(Enum):
        PELVIS = 0
        SPINE_NAVEL = 1
        SPINE_CHEST = 2
        NECK = 3
        CLAVICLE_LEFT = 4
        SHOULDER_LEFT = 5
        ELBOW_LEFT = 6
        WRIST_LEFT = 7
        HAND_LEFT = 8
        HANDTIP_LEFT = 9
        THUMB_LEFT = 10
        CLAVICLE_RIGHT = 11
        SHOULDER_RIGHT = 12
        ELBOW_RIGHT = 13
        WRIST_RIGHT = 14
        HAND_RIGHT = 15
        HANDTIP_RIGHT = 16
        THUMB_RIGHT = 17
        HIP_LEFT = 18
        KNEE_LEFT = 19
        ANKLE_LEFT = 20
        FOOT_LEFT = 21
        HIP_RIGHT = 22
        KNEE_RIGHT = 23
        ANKLE_RIGHT = 24
        FOOT_RIGHT = 25
        HEAD = 26
        NOSE = 27
        EYE_LEFT = 28
        EAR_LEFT = 29
        EYE_RIGHT = 30
        EAR_RIGHT = 31

class BodyCategory(Enum):
        HEAD = 0
        RIGHT_ARM = 1
        RIGHT_HAND = 7
        LEFT_ARM = 2
        LEFT_HAND = 6
        TORSO = 3
        RIGHT_LEG = 4
        LEFT_LEG = 5

def getPointSubcategory(joint):
     if(joint == Joint.PELVIS or joint == Joint.NECK or joint == Joint.SPINE_NAVEL or joint == Joint.SPINE_CHEST):
          return BodyCategory.TORSO
     if(joint == Joint.WRIST_LEFT or joint == Joint.CLAVICLE_LEFT or joint == Joint.SHOULDER_LEFT or joint == Joint.ELBOW_LEFT):
           return BodyCategory.LEFT_ARM
     if(joint == Joint.HAND_LEFT or joint == Joint.HANDTIP_LEFT or joint == Joint.THUMB_LEFT):
          return BodyCategory.LEFT_HAND
     if(joint == Joint.WRIST_RIGHT or joint == Joint.CLAVICLE_RIGHT or joint == Joint.SHOULDER_RIGHT or joint == Joint.ELBOW_RIGHT):
        return BodyCategory.RIGHT_ARM
     if(joint == Joint.HAND_RIGHT or joint == Joint.HANDTIP_RIGHT or joint == Joint.THUMB_RIGHT):
          return BodyCategory.RIGHT_HAND
     if(joint == Joint.HIP_LEFT or joint == Joint.KNEE_LEFT or joint == Joint.ANKLE_LEFT or joint == Joint.FOOT_LEFT):
          return BodyCategory.LEFT_LEG
     if(joint == Joint.HIP_RIGHT or joint == Joint.KNEE_RIGHT or joint == Joint.ANKLE_RIGHT or joint == Joint.FOOT_RIGHT):
          return BodyCategory.RIGHT_LEG
     if(joint == Joint.HEAD or joint == Joint.NOSE or joint == Joint.EYE_LEFT 
        or joint == Joint.EAR_LEFT or joint == Joint.EYE_RIGHT or joint == Joint.EAR_RIGHT):
          return BodyCategory.HEAD

bone_list = [
        [
            Joint.SPINE_CHEST, 
            Joint.SPINE_NAVEL
        ],
        [
            Joint.SPINE_NAVEL,
            Joint.PELVIS
        ],
        [
            Joint.SPINE_CHEST,
            Joint.NECK
        ],
        [
            Joint.NECK,
            Joint.HEAD
        ],
        [
            Joint.HEAD,
            Joint.NOSE
        ],
        [
            Joint.SPINE_CHEST,
            Joint.CLAVICLE_LEFT
        ],
        [
            Joint.CLAVICLE_LEFT,
            Joint.SHOULDER_LEFT
        ],
        [
            Joint.SHOULDER_LEFT,
            Joint.ELBOW_LEFT
        ],
        [
            Joint.ELBOW_LEFT,
            Joint.WRIST_LEFT
        ],
        [
            Joint.NOSE,
            Joint.EYE_LEFT
        ],
        [
            Joint.EYE_LEFT,
            Joint.EAR_LEFT
        ],
        [
            Joint.SPINE_CHEST,
            Joint.CLAVICLE_RIGHT
        ],
        [
            Joint.CLAVICLE_RIGHT,
            Joint.SHOULDER_RIGHT
        ],
        [
            Joint.SHOULDER_RIGHT,
            Joint.ELBOW_RIGHT
        ],
        [
            Joint.ELBOW_RIGHT,
            Joint.WRIST_RIGHT
        ],
        [
            Joint.NOSE,
            Joint.EYE_RIGHT
        ],
        [
            Joint.EYE_RIGHT,
            Joint.EAR_RIGHT
        ]
]

parser = argparse.ArgumentParser()
parser.add_argument('--maxHands', nargs='?', default=6)
parser.add_argument('--minDetectionConfidence', nargs='?', default=0.6)
parser.add_argument('--minTrackingConfidence', nargs='?', default=0.6)
parser.add_argument('--videoPath', nargs='?', default="F:\\Weights_Task\\Data\\Fib_weights_original_videos\\Group_02-master.mkv")
parser.add_argument('--jsonPath', nargs='?', default="F:\\Weights_Task\\Data\\Group_02-master.json")
parser.add_argument('--initialFrame', nargs='?', default=5000) #start counting from frame 0 because opencv is zero based

args = parser.parse_args()

# video file
cap = cv2.VideoCapture(args.videoPath)
cap.set(cv2.CAP_PROP_POS_FRAMES, args.initialFrame) # this is zero based
jsonFile = open(args.jsonPath)

# read in azure skeleton data
skeletonData = json.load(jsonFile)
frameData = skeletonData["frames"]
_, rotation, translation, dist = utils.getCalibrationFromFile(skeletonData["camera_calibration"])
cameraMatrix = utils.getMasterCameraMatrix()

#BGR
# Red, Green, Orange, Blue, Purple
colors = [(0, 0, 255), (0, 255, 0), (0, 140, 255), (255, 0, 0), (139,34,104)]
dotColors = [(0, 0, 139), (20,128,48), (71,130,170), (205,95,58), (205,150,205)]

frameCount = 0
shift = 7

mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands = mpHands.Hands(max_num_hands=args.maxHands, min_detection_confidence=args.minDetectionConfidence, min_tracking_confidence=args.minTrackingConfidence)

while cap.isOpened():
    success, frame = cap.read()
    h, w, c = frame.shape
    if not success:
        print("Ignoring empty camera frame.")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # result_hands = hands.process(framergb)
    # if result_hands.multi_hand_landmarks:
    #     landmarks = []
    #     for index, handslms in enumerate(result_hands.multi_hand_landmarks):
    #         for lm in handslms.landmark:
    #             # print(id, lm)
    #             lmx = int(lm.x * w)
    #             lmy = int(lm.y * h)

    #             landmarks.append([lmx, lmy])

    #         # Drawing landmarks on frames
    #         mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

    bodies = frameData[frameCount + args.initialFrame]["bodies"]
    for bodyIndex, body in enumerate(bodies):  
        bodyId = int(body["body_id"])
        dotColor = dotColors[bodyId % len(dotColors)]; 
        color = colors[bodyId % len(colors)]; 
        dictionary = {}
        for jointIndex, joint in enumerate(body["joint_positions"]):
            bodyLocation = getPointSubcategory(Joint(jointIndex))
            print(f"{bodyId}")
            if(bodyLocation != BodyCategory.RIGHT_LEG and bodyLocation != BodyCategory.LEFT_LEG
            and bodyLocation != BodyCategory.RIGHT_HAND and bodyLocation != BodyCategory.LEFT_HAND):
                points2D, _ = cv2.projectPoints(
                    np.array(joint), 
                    rotation,
                    translation,
                    cameraMatrix,
                    dist)  
                
                point = (int(points2D[0][0][0] * 2**shift),int(points2D[0][0][1] * 2**shift))
                dictionary[Joint(jointIndex)] = point
                cv2.circle(frame, point, radius=15, color=dotColor, thickness=15, shift=shift)
        for bone in bone_list:
            if(getPointSubcategory(bone[0]) == BodyCategory.RIGHT_ARM or getPointSubcategory(bone[1]) == BodyCategory.RIGHT_ARM):
                cv2.line(frame, dictionary[bone[0]], dictionary[bone[1]], color=(255,255,255), thickness=3, shift=shift)
            else:
                cv2.line(frame, dictionary[bone[0]], dictionary[bone[1]], color=color, thickness=3, shift=shift)
        cv2.putText(frame, str(bodyId), (50, 100 + (50 * bodyIndex)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                

    cv2.putText(frame, "Frame: " + str(frameCount + args.initialFrame), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    # cv2.imwrite("C:\\Users\\vanderh\\Desktop\\Paper\\" + str(frameCount + args.initialFrame) + "sub2.png", frame) 
    frame = cv2.resize(frame, (960, 540))
    cv2.imshow("Frame", frame)
    frameCount+=1
    
    if cv2.waitKey(5) == ord('q'):
        break
       
# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()
