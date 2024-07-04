from featureModules.IFeature import *
import mediapipe as mp
import joblib
import torch.nn as nn
import os
import torch
from utils import *

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

class PoseFeature(IFeature):
    def __init__(self):
        #  required arguments
        input_size = 224
        hidden_size = 300
        output_size = 1
        
        # initialize a model 
        self.leftModel = SkeletonPoseClassifier(input_size = input_size,hidden_size=hidden_size,output_size=output_size)
        self.leftModel.load_state_dict(torch.load(".\\featureModules\\pose\\poseModels\\skeleton_pose_classifier_left.pt"))
        self.leftModel.eval()

        self.middleModel = SkeletonPoseClassifier(input_size = input_size,hidden_size=hidden_size,output_size=output_size)
        self.middleModel.load_state_dict(torch.load(".\\featureModules\\pose\\poseModels\\skeleton_pose_classifier_middle.pt"))
        self.middleModel.eval()

        self.rightModel = SkeletonPoseClassifier(input_size = input_size,hidden_size=hidden_size,output_size=output_size)
        self.rightModel.load_state_dict(torch.load(".\\featureModules\\pose\\poseModels\\skeleton_pose_classifier_right.pt"))
        self.rightModel.eval()

    def processFrame(self, bodies, frame):
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
                poseModel = self.leftModel
                body = b
                position = "left"
            elif x > left_position and x < middle_position:
                #print("middle")
                poseModel = self.middleModel
                body = b
                position = "middle"
            else:
                #print("right")
                poseModel = self.rightModel
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
            del o, p
            torch.cuda.empty_cache()
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