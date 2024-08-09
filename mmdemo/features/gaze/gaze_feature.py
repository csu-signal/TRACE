from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import BodyTrackingInterface, GazeInterface

# import helpers
# from mmdemo.features.proposition.helpers import ...


@final
class Gaze(BaseFeature):
    # LOG_FILE = "gazeOutput.csv"

    @classmethod
    def get_input_interfaces(cls):
        return [BodyTrackingInterface]

    @classmethod
    def get_output_interface(cls):
        return GazeInterface

    def initialize(self):
        # self.shift = shift
        # self.faceDetector = MTCNN()

        # model_dir = str(Path(__file__).parent / "gazeDetectionModels" / "Model" / "1")
        # self.gazeModel = keras.models.load_model(
        #     model_dir,
        #     custom_objects={
        #         "euclideanLoss": euclideanLoss,
        #         "categorical_accuracy": categorical_accuracy,
        #     },
        # )

        # if log_dir is not None:
        #     self.logger = Logger(file=log_dir / self.LOG_FILE)
        # else:
        #     self.logger = Logger()

        # self.logger.write_csv_headers("frame_index", "bodyId", "targets")
        pass

    def get_output(self, t: BodyTrackingInterface):
        if not t.is_new():
            return None

        # call __, create interface, and return

        #       #faces,heads,images=load_frame(frame,framergb,self.faceDetector,shift)
        # faces,heads,images,bodyIds=load_frame_azure(frame,framergb,bodies, rotation, translation, cameraMatrix, dist, self.shift)
        # targets = []
        # if(len(faces) > 0):
        #     preds = predict_gaze(self.gazeModel, images, faces, heads)
        #     gazeCount[0] += 1
        #     for index, head in enumerate(heads):
        #         key = bodyIds[index]
        #         #print(key)
        #         if key not in gazeHead:
        #             gazeHead[key] = []
        #             gazePred[key] = []

        #         gazeHead[key].append([(heads[index][0] * w), (heads[index][1] * h)])
        #         gazePred[key].append([(preds[0][index][0] * w), (preds[0][index][1] * h)])

        #         if(len(gazePred[key]) == 5):
        #             sumx = 0
        #             sumy = 0
        #             for point in gazeHead[key]:
        #                 sumx += point[0]
        #                 sumy += point[1]

        #             headX_average = int(sumx / 5)
        #             headY_average = int(sumy / 5)

        #             sumx = 0
        #             sumy = 0
        #             for point in gazePred[key]:
        #                 sumx += point[0]
        #                 sumy += point[1]

        #             predX_average = int(sumx / 5)
        #             predY_average = int(sumy / 5)

        #             lenAB = math.sqrt(pow(headX_average - predX_average, 2.0) + pow(headY_average - predY_average, 2.0))
        #             #print(lenAB)

        #             length = 500
        #             if(lenAB < length):
        #                 # print("Made it into the length update")
        #                 # print("Before")
        #                 # print(predX_average)
        #                 # print(predY_average)
        #                 unitSlopeX = (predX_average-headX_average) / lenAB
        #                 unitSlopeY = (predY_average-headY_average) / lenAB

        #                 predX_average = int(headX_average + (unitSlopeX * length))
        #                 predY_average = int(headY_average + (unitSlopeY * length))
        #                 # print("After")
        #                 # print(predX_average)
        #                 # print(predY_average)

        #                 # predX_average = int(predX_average + (predX_average - headX_average) / lenAB * (length - lenAB))
        #                 # predY_average = int(predY_average + (predY_average - predX_average) / lenAB * (length - lenAB))

        #             head3D, h_Success = convertTo3D(cameraMatrix, dist, depth, headX_average, headY_average)
        #             pred3D, p_Success = convertTo3D(cameraMatrix, dist, depth, predX_average, predY_average)

        #             if(h_Success == ParseResult.Success and p_Success == ParseResult.Success):
        #                 pred_p1 = int((predX_average) * 2**self.shift)
        #                 pred_p2 = int((predY_average) * 2**self.shift)

        #                 head_p1 = int((headX_average) * 2**self.shift)
        #                 head_p2 = int((headY_average) * 2**self.shift)
        #                 cv2.line(frame, (head_p1, head_p2), (pred_p1, pred_p2), thickness=5, shift=self.shift, color=(255, 107, 170))

        #                 cone = ConeShape(head3D, pred3D, 80, 100, cameraMatrix, dist)
        #                 cone.projectRadiusLines(self.shift, frame, False, False, True)

        #                 targets = checkBlocks(blocks, blockStatus, cameraMatrix, dist, depth, cone, frame, self.shift, True)

        #             # for key in gazeHead:
        #             #     print(key)
        #             copyHead = gazeHead[key]
        #             #print(copyHead)

        #             gazeHead[key] = []
        #             gazeHead[key].append(copyHead[1])
        #             gazeHead[key].append(copyHead[2])
        #             gazeHead[key].append(copyHead[3])
        #             gazeHead[key].append(copyHead[4])

        #             copyPred = gazePred[key]
        #             gazePred[key] = []
        #             gazePred[key].append(copyPred[1])
        #             gazePred[key].append(copyPred[2])
        #             gazePred[key].append(copyPred[3])
        #             gazePred[key].append(copyPred[4])
        #             del head3D, h_Success, pred3D, p_Success
        #             keras.backend.clear_session()
        #             gc.collect()

        #         descriptions = []
        #         for t in targets:
        #             descriptions.append(t.description)

        #         self.logger.append_csv(frameIndex, key, targets)
