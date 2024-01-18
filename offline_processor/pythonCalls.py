import math
import cv2
import mediapipe as mp
import os
import shutil

if os.path.exists("./Camera1"):
        shutil.rmtree("./Camera1")
os.makedirs("./Camera1")

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=6, min_detection_confidence=0.2)
mpDraw = mp.solutions.drawing_utils

def myabs(x):
    return math.fabs(x)

def openFrame(data):
    try:
        encoding = 'utf-8'
        path = data.decode(encoding)   
        print(path)

        frame = cv2.imread(path)
        x , y, c = frame.shape

        # Flip the frame vertically
        frame = cv2.flip(frame, 1)

        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
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
                    mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

        # Show the final output
        frame = cv2.resize(frame, (960, 640))
        cv2.imshow("Output", frame)
        cv2.waitKey(250)
        cv2.destroyAllWindows()
        return 1
    except Exception as e:
        print(e)
