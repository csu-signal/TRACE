from typing import final
from pathlib import Path
import joblib
import cv2
import mediapipe as mp

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import (  # GestureInterface,
    BodyTrackingInterface,
    CameraCalibrationInterface,
    ColorImageInterface,
    DepthImageInterface,
    GestureConesInterface,
)
from mmdemo.utils.cone_shape import ConeShape
from mmdemo.interfaces.data import Cone
from mmdemo.utils.hand_utils import createBoundingBox, getAverageHandLocations, processHands
from mmdemo.utils.point_vector_logic import processPoint
from mmdemo.utils.support_utils import Handedness, ParseResult

# import helpers
# from mmdemo.features.proposition.helpers import ...

@final
class Gesture(BaseFeature[GestureConesInterface]):
    # LOG_FILE = "gestureOutput.csv"

    @classmethod
    def get_input_interfaces(cls):
        return [
            ColorImageInterface,
            DepthImageInterface,
            BodyTrackingInterface,
            CameraCalibrationInterface,
        ]

    @classmethod
    def get_output_interface(cls):
        return GestureConesInterface

    def initialize(self):
        model_path = Path(__file__).parent / "bestModel-pointing.pkl"
        self.loaded_model = joblib.load(str(model_path))
            
        mpHands = mp.solutions.hands
        self.hands = mpHands.Hands(max_num_hands=1, static_image_mode= True, min_detection_confidence=0.4, min_tracking_confidence=0)

        # self.devicePoints = {}
        # self.shift = shift
        # self.blockCache = {}

        # self.init_logger(log_dir)
        pass

    def get_output(
        self,
        col: ColorImageInterface,
        depth: DepthImageInterface,
        bt: BodyTrackingInterface,
        cal: CameraCalibrationInterface
    ):
        if not col.is_new() or not depth.is_new() or not bt.is_new() or not cal.is_new():
            return None
        
        cones = []
        body_ids = []
        handedness = []
        h, w, c = col.frame.shape
        framergb=cv2.cvtColor(col.frame, cv2.COLOR_BGR2RGB)
        for _, body in enumerate(bt.bodies):
            leftXAverage, leftYAverage, rightXAverage, rightYAverage = getAverageHandLocations(
                body, w, h, cal.rotation, cal.translation, cal.cameraMatrix, cal.distortion)
            rightBox = createBoundingBox(rightXAverage, rightYAverage)
            leftBox = createBoundingBox(leftXAverage, leftYAverage)
            leftCone = self.findCones(
                col.frame,
                framergb,
                Handedness.Left,
                leftBox,
                cal.cameraMatrix,
                cal.distortion,
                depth
                )
            
            if(leftCone != None):
                cones.append(leftCone)
                body_ids.append(int(body['body_id']))
                handedness.append(Handedness.Left)

            rightCone = self.findCones(
                col.frame,
                framergb,
                Handedness.Right,
                rightBox,
                cal.cameraMatrix,
                cal.distortion,
                depth
                )
            
            if(rightCone != None):
                cones.append(rightCone)
                body_ids.append(int(body['body_id']))
                handedness.append(Handedness.Right)
            
        return GestureConesInterface(body_ids=body_ids, handedness=handedness, cones=cones)
        # call __, create interface, and return

    # TODO logging
    # def log_gesture(self, frame: int, descriptions: list[str], body_id, handedness: str):
    #     self.logger.append_csv(frame, json.dumps(descriptions), body_id, handedness)

    # TODO rendering logic
    #     for key in self.devicePoints:
    #         if(key == deviceId):
    #             if includeText:
    #                 if(len(self.devicePoints[key]) == 0):
    #                     cv2.putText(frame, "NO POINTS", (50,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    #                 else:
    #                     cv2.putText(frame, "POINTS DETECTED", (50,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    #                     pointsFound = True
    #             for hand in self.devicePoints[key]:
    #                 for point in hand:
    #                     cv2.circle(frame, point, radius=2, thickness= 2, color=(0,255,0))

    def findCones(self, frame, framergb, handedness, box, cameraMatrix, dist, depth):
        ## to help visualize where the hand localization is focused
        # dotColor = dotColors[bodyId % len(dotColors)]
        # cv2.rectangle(frame,
        #     (box[0]* 2**self.shift, box[1]* 2**self.shift),
        #     (box[2]* 2**self.shift, box[3]* 2**self.shift),
        #     color=dotColor,
        #     thickness=3,
        #     shift=self.shift)

        #media pipe on the sub box only
        frameBox = framergb[box[1]:box[3], box[0]:box[2]]
        h, w, c = frameBox.shape

        if(h > 0 and w > 0):
            results = self.hands.process(frameBox)

            handedness_result = results.multi_handedness
            if results.multi_hand_landmarks:
                for index, handslms in enumerate(results.multi_hand_landmarks):
                    returned_handedness = handedness_result[index].classification[0]
                    if(returned_handedness.label == handedness.value):
                        normalized = processHands(frame, handslms)
                        prediction = self.loaded_model.predict_proba([normalized])
                        #print(prediction)

                        if prediction[0][0] >= 0.3:
                            landmarks = []
                            for lm in handslms.landmark:
                                lmx, lmy = int(lm.x * w) + box[0], int(lm.y * h) + box[1]
                                #cv2.circle(frame, (lmx, lmy), radius=2, thickness= 2, color=(0,0,255))
                                landmarks.append([lmx, lmy])

                            tx, ty, mediaPipe8, bx, by, mediaPipe5, nextPoint, success = processPoint(landmarks, box, w, h, cameraMatrix, dist, depth)
                            #cv2.putText(frame, str(success), (50,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)

                            if success == ParseResult.Success:
                                return ConeShape(mediaPipe5, nextPoint, 40, 70, cameraMatrix, dist)
                                #return Cone(mediaPipe5, nextPoint, 40, 70)
                            else:
                                return None
                            
                                # TODO move to rendering feature
                                # point8_2D = convert2D(mediaPipe8, cameraMatrix, dist)
                                # point5_2D = convert2D(mediaPipe5, cameraMatrix, dist)
                                # pointExtended_2D = convert2D(nextPoint, cameraMatrix, dist)

                                # point_8 = (int(point8_2D[0] * 2**self.shift),int(point8_2D[1] * 2**self.shift))
                                # point_5 = (int(point5_2D[0] * 2**self.shift),int(point5_2D[1] * 2**self.shift))
                                # point_Extended = (int(pointExtended_2D[0] * 2**self.shift),int(pointExtended_2D[1] * 2**self.shift))

                                # cv2.line(frame, point_8, point_Extended, color=(0, 165, 255), thickness=5, shift=self.shift)
                                # cv2.line(frame, point_5, point_8, color=(0, 165, 255), thickness=5, shift=self.shift)

                                # cv2.circle(frame, point_8, radius=15, color=(255,0,0), thickness=15, shift=self.shift)
                                # cv2.circle(frame, (int(tx), int(ty)), radius=4, color=(0, 0, 255), thickness=-1)
                                # cv2.circle(frame, point_5, radius=15, color=(255,0,0), thickness=15, shift=self.shift)
                                # cv2.circle(frame, (int(bx), int(by)), radius=4, color=(0, 0, 255), thickness=-1)
                                # cv2.circle(frame, point_Extended, radius=15, color=(255,0,0), thickness=15, shift=self.shift)
                                #cone.projectRadiusLines(self.shift, frame, True, False, False)

                                ## TODO move to target checking features
                                # targets = checkBlocks(blocks, blockStatus, cameraMatrix, dist, depth, cone, frame, self.shift, False, gesture=True, index=mediaPipe8)
                                # frame_bin = get_frame_bin(frameIndex)
                                # if(targets):
                                #     self.blockCache[frame_bin] = [t.description for t in targets]

                                # descriptions = []
                                # for t in targets:
                                #     descriptions.append(t.description)

                                # self.log_gesture(frameIndex, [d.value for d in descriptions], bodyId, handedness.value)
