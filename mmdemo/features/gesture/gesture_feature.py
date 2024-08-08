from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import GestureInterface, TranscriptionInterface

# import helpers
# from mmdemo.features.proposition.helpers import ...


@final
class Gesture(BaseFeature):
    def __init__(self, *args):
        super().__init__()
        self.register_dependencies([TranscriptionInterface], args)

    @classmethod
    def get_output_interface(cls):
        return GestureInterface

    def initialize(self):
        # initialize prop model
        pass

    def get_output(self, t: TranscriptionInterface):
        if not t.is_new():
            return None

        # call __, create interface, and return

    # def log_gesture(self, frame: int, descriptions: list[str], body_id, handedness: str):
    #     self.logger.append_csv(frame, json.dumps(descriptions), body_id, handedness)

    # def processFrame(self, deviceId, bodies, w, h, rotation, translation, cameraMatrix, dist, frame, framergb, depth, blocks, blockStatus, frameIndex, includeText):
    #     points = []
    #     pointsFound = False
    #     for _, body in enumerate(bodies):
    #         leftXAverage, leftYAverage, rightXAverage, rightYAverage = getAverageHandLocations(body, w, h, rotation, translation, cameraMatrix, dist)
    #         rightBox = createBoundingBox(rightXAverage, rightYAverage)
    #         leftBox = createBoundingBox(leftXAverage, leftYAverage)
    #         self.findHands(frame,
    #                 framergb,
    #                 int(body['body_id']),
    #                 Handedness.Left,
    #                 leftBox,
    #                 points,
    #                 cameraMatrix,
    #                 dist,
    #                 depth,
    #                 blocks,
    #                 blockStatus,
    #                 frameIndex
    #                 )
    #         self.findHands(frame,
    #                 framergb,
    #                 int(body['body_id']),
    #                 Handedness.Right,
    #                 rightBox,
    #                 points,
    #                 cameraMatrix,
    #                 dist,
    #                 depth,
    #                 blocks,
    #                 blockStatus,
    #                 frameIndex
    #                 )
    #         self.devicePoints[deviceId] = points

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

    # def findHands(self, frame, framergb, bodyId, handedness, box, points, cameraMatrix, dist, depth, blocks, blockStatus, frameIndex):
    #     ## to help visualize where the hand localization is focused
    #     # dotColor = dotColors[bodyId % len(dotColors)]
    #     # cv2.rectangle(frame,
    #     #     (box[0]* 2**self.shift, box[1]* 2**self.shift),
    #     #     (box[2]* 2**self.shift, box[3]* 2**self.shift),
    #     #     color=dotColor,
    #     #     thickness=3,
    #     #     shift=self.shift)

    #     #media pipe on the sub box only
    #     frameBox = framergb[box[1]:box[3], box[0]:box[2]]
    #     h, w, c = frameBox.shape

    #     if(h > 0 and w > 0):
    #         results = hands.process(frameBox)

    #         handedness_result = results.multi_handedness
    #         if results.multi_hand_landmarks:
    #             for index, handslms in enumerate(results.multi_hand_landmarks):
    #                 returned_handedness = handedness_result[index].classification[0]
    #                 if(returned_handedness.label == handedness.value):
    #                     normalized = processHands(frame, handslms)
    #                     prediction = self.loaded_model.predict_proba([normalized])
    #                     #print(prediction)

    #                     if prediction[0][0] >= 0.3:
    #                         landmarks = []
    #                         for lm in handslms.landmark:
    #                             lmx, lmy = int(lm.x * w) + box[0], int(lm.y * h) + box[1]
    #                             #cv2.circle(frame, (lmx, lmy), radius=2, thickness= 2, color=(0,0,255))
    #                             landmarks.append([lmx, lmy])

    #                         points.append(landmarks)
    #                         tx, ty, mediaPipe8, bx, by, mediaPipe5, nextPoint, success = processPoint(landmarks, box, w, h, cameraMatrix, dist, depth)
    #                         #cv2.putText(frame, str(success), (50,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)

    #                         if success == ParseResult.Success:
    #                             point8_2D = convert2D(mediaPipe8, cameraMatrix, dist)
    #                             point5_2D = convert2D(mediaPipe5, cameraMatrix, dist)
    #                             pointExtended_2D = convert2D(nextPoint, cameraMatrix, dist)

    #                             point_8 = (int(point8_2D[0] * 2**self.shift),int(point8_2D[1] * 2**self.shift))
    #                             point_5 = (int(point5_2D[0] * 2**self.shift),int(point5_2D[1] * 2**self.shift))
    #                             point_Extended = (int(pointExtended_2D[0] * 2**self.shift),int(pointExtended_2D[1] * 2**self.shift))

    #                             cv2.line(frame, point_8, point_Extended, color=(0, 165, 255), thickness=5, shift=self.shift)
    #                             cv2.line(frame, point_5, point_8, color=(0, 165, 255), thickness=5, shift=self.shift)

    #                             cv2.circle(frame, point_8, radius=15, color=(255,0,0), thickness=15, shift=self.shift)
    #                             cv2.circle(frame, (int(tx), int(ty)), radius=4, color=(0, 0, 255), thickness=-1)
    #                             cv2.circle(frame, point_5, radius=15, color=(255,0,0), thickness=15, shift=self.shift)
    #                             cv2.circle(frame, (int(bx), int(by)), radius=4, color=(0, 0, 255), thickness=-1)
    #                             cv2.circle(frame, point_Extended, radius=15, color=(255,0,0), thickness=15, shift=self.shift)

    #                             cone = ConeShape(mediaPipe5, nextPoint, 40, 70, cameraMatrix, dist)
    #                             cone.projectRadiusLines(self.shift, frame, True, False, False)

    #                             ## TODO keep track of participant?
    #                             targets = checkBlocks(blocks, blockStatus, cameraMatrix, dist, depth, cone, frame, self.shift, False, gesture=True, index=mediaPipe8)
    #                             frame_bin = get_frame_bin(frameIndex)
    #                             if(targets):
    #                                 self.blockCache[frame_bin] = [t.description for t in targets]

    #                             descriptions = []
    #                             for t in targets:
    #                                 descriptions.append(t.description)

    #                             self.log_gesture(frameIndex, [d.value for d in descriptions], bodyId, handedness.value)
