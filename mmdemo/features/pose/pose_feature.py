from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import BodyTrackingInterface, PoseInterface

# import helpers
# from mmdemo.features.proposition.helpers import ...


@final
class Pose(BaseFeature[PoseInterface]):
    """
    Detect the pose of participants

    Input interface is `BodyTrackingInterface`

    Output interface is `PoseInterface`
    """

    def __init__(self, bt: BaseFeature[BodyTrackingInterface]):
        super().__init__(bt)

    def initialize(self):
        # #  required arguments
        # input_size = 224
        # hidden_size = 300
        # output_size = 1

        # model_dir = Path(__file__).parent / "poseModels"

        # # initialize a model
        # self.leftModel = SkeletonPoseClassifier(input_size = input_size,hidden_size=hidden_size,output_size=output_size)
        # self.leftModel.load_state_dict(torch.load(str(model_dir / "skeleton_pose_classifier_left.pt")))
        # self.leftModel.eval()

        # self.middleModel = SkeletonPoseClassifier(input_size = input_size,hidden_size=hidden_size,output_size=output_size)
        # self.middleModel.load_state_dict(torch.load(str(model_dir / "skeleton_pose_classifier_middle.pt")))
        # self.middleModel.eval()

        # self.rightModel = SkeletonPoseClassifier(input_size = input_size,hidden_size=hidden_size,output_size=output_size)
        # self.rightModel.load_state_dict(torch.load(str(model_dir / "skeleton_pose_classifier_right.pt")))
        # self.rightModel.eval()

        # if log_dir is not None:
        #     self.logger = Logger(file=log_dir / self.LOG_FILE)
        # else:
        #     self.logger = Logger()

        # self.logger.write_csv_headers("frame_index", "participant", "engagement")
        pass

    def get_output(self, bt: BodyTrackingInterface):
        if not bt.is_new():
            return None

        # call _, create interface, and return

    # def processFrame(self, bodies, frame, frameIndex, includeText):
    #     left_position = -400
    #     middle_position = 400

    #     # left_position = 800
    #     # middle_position = 1200

    #     # cv2.circle(frame, (int(left_position), 800), radius=15, color=(255,0,0), thickness=15)
    #     # cv2.circle(frame, (int(middle_position), 800), radius=15, color=(255,0,0), thickness=15)

    #     for b in bodies:
    #         # points2D, _ = cv2.projectPoints(
    #         #         np.array(b['joint_positions'][1]),
    #         #         rotation,
    #         #         translation,
    #         #         cameraMatrix,
    #         #         dist)
    #         #x = points2D[0][0][0]
    #         x = b['joint_positions'][1][0]
    #         #print(x)

    #         if x < left_position:
    #            # print("left")
    #             poseModel = self.leftModel
    #             body = b
    #             position = "left"
    #         elif x > left_position and x < middle_position:
    #             #print("middle")
    #             poseModel = self.middleModel
    #             body = b
    #             position = "middle"
    #         else:
    #             #print("right")
    #             poseModel = self.rightModel
    #             body = b
    #             position = "right"

    #         # print(b['joint_positions'][1][0])
    #         # if b['joint_positions'][1][0] > left_position:
    #         #     print("left")
    #         #     poseModel = leftModel
    #         #     body = b
    #         #     position = "left"
    #         # elif b['joint_positions'][1][0] < middle_position:
    #         #     print("middle")
    #         #     poseModel = middleModel
    #         #     body = b
    #         #     position = "middle"
    #         # else:
    #         #     print("right")
    #         #     poseModel = rightModel
    #         #     body = b
    #         #     position = "right"

    #         tensors = []
    #         orientation_data = body['joint_orientations']
    #         position_data = body['joint_positions']
    #         o = torch.tensor(orientation_data).flatten()
    #         p = torch.tensor(position_data).flatten() / 1000 # normalize to scale of orientations
    #         tensors.append(torch.concat([o, p])) # concatenating orientation to position
    #         del o, p
    #         torch.cuda.empty_cache()
    #         output = poseModel(torch.stack(tensors))
    #         # prediction = int(torch.argmax(output))
    #         prediction = output.detach().numpy()[0][0] > 0.5

    #         # print("Prediction: " + str(prediction))
    #         # print("Output: " + str(output))

    #         engagement = "leaning out" if prediction == 0 else "leaning in"
    #         color = (255,0,0) if prediction == 0 else (39,142,37)
    #         if position == "left":
    #             if includeText:
    #                 cv2.putText(frame, "P1: " + engagement, (50,200), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    #             self.logger.append_csv(frameIndex, "P1", engagement)
    #         elif position == "middle":
    #             if includeText:
    #                 cv2.putText(frame, "P2: " + engagement, (50,250), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    #             self.logger.append_csv(frameIndex, "P2", engagement)
    #         else:
    #             if includeText:
    #                 cv2.putText(frame, "P3: " + engagement, (50,300), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    #             self.logger.append_csv(frameIndex, "P3", engagement)
