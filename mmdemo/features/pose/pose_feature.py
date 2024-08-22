from typing import final
from pathlib import Path
from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import BodyTrackingInterface, PoseInterface
import torch.nn as nn
import torch

# import helpers
# from mmdemo.features.proposition.helpers import ...

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

@final
class Pose(BaseFeature[PoseInterface]):
    """
    Detect the pose of participants

    Input interface is `BodyTrackingInterface`

    Output interface is `PoseInterface`
    """

    def __init__(self, bt: BaseFeature[BodyTrackingInterface]):
        super().__init__(bt)

    def initialize(self, leftPosition, middlePosition):
        self.left_position = leftPosition
        self.middle_position = middlePosition

        #  required arguments
        input_size = 224
        hidden_size = 300
        output_size = 1

        model_dir = Path(__file__).parent / "poseModels"

        # initialize a model
        self.leftModel = SkeletonPoseClassifier(input_size = input_size,hidden_size=hidden_size,output_size=output_size)
        self.leftModel.load_state_dict(torch.load(str(model_dir / "skeleton_pose_classifier_left.pt")))
        self.leftModel.eval()

        self.middleModel = SkeletonPoseClassifier(input_size = input_size,hidden_size=hidden_size,output_size=output_size)
        self.middleModel.load_state_dict(torch.load(str(model_dir / "skeleton_pose_classifier_middle.pt")))
        self.middleModel.eval()

        self.rightModel = SkeletonPoseClassifier(input_size = input_size,hidden_size=hidden_size,output_size=output_size)
        self.rightModel.load_state_dict(torch.load(str(model_dir / "skeleton_pose_classifier_right.pt")))
        self.rightModel.eval()

        # if log_dir is not None:
        #     self.logger = Logger(file=log_dir / self.LOG_FILE)
        # else:
        #     self.logger = Logger()

        # self.logger.write_csv_headers("frame_index", "participant", "engagement")
        # pass

    def get_output(self, bt: BodyTrackingInterface):
        if not bt.is_new():
            return None

        poses = []
        for b in bt.bodies:
            x = b['joint_positions'][1][0]
            if x < self.left_position:
               # print("left")
                poseModel = self.leftModel
                body = b
                position = "left"
            elif x > self.left_position and x < self.middle_position:
                #print("middle")
                poseModel = self.middleModel
                body = b
                position = "middle"
            else:
                #print("right")
                poseModel = self.rightModel
                body = b
                position = "right"

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

            engagement = "leaning out" if prediction == 0 else "leaning in"
            if position == "left":
                poses.append(["P1", engagement])
            elif position == "middle":
                poses.append(["P2", engagement])
            else:
                poses.append(["P3", engagement])
        return PoseInterface(poses=poses)
