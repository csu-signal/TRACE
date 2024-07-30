import cv2 as cv
import numpy as np

from featureModules.IFeature import *
from logger import Logger
from utils import ConeShape, Joint, checkBlocks, convert2D


class GazeBodyTrackingFeature(IFeature):
    LOG_FILE = "gazeOutput.csv"

    def __init__(self, shift, log_dir=None):
        self.shift = shift

        if log_dir is not None:
            self.logger = Logger(file=log_dir / self.LOG_FILE)
        else:
            self.logger = Logger()

        self.logger.write_csv_headers("frame_index", "bodyId", "targets")

    def world_to_camera_coords(self, r_w, rotation, translation):
        return np.dot(rotation, r_w) + translation

    def get_joint(self, joint, body, rotation, translation):
        r_w = np.array(body["joint_positions"][joint.value])
        return self.world_to_camera_coords(r_w, rotation, translation)

    def processFrame(self, bodies, w, h, rotation, translation, cameraMatrix, dist, frame, framergb, depth, blocks, blockStatus, frame_count):
        for b in bodies:
            body_id = b["body_id"]

            nose = self.get_joint(Joint.NOSE, b, rotation, translation)

            ear_left = self.get_joint(Joint.EAR_LEFT, b, rotation, translation)
            ear_right = self.get_joint(Joint.EAR_RIGHT, b, rotation, translation)
            ear_center = (ear_left + ear_right) / 2

            eye_left = self.get_joint(Joint.EYE_LEFT, b, rotation, translation)
            eye_right = self.get_joint(Joint.EYE_RIGHT, b, rotation, translation)

            dir = nose - ear_center
            dir /= np.linalg.norm(nose - ear_center)

            origin = (eye_left + eye_right + nose) / 3

            p1_3d = origin
            p2_3d = origin + 1000*dir

            cone = ConeShape(p1_3d, p2_3d, 80, 100, cameraMatrix, dist)
            cone.projectRadiusLines(self.shift, frame, False, False, True)

            p1 = convert2D(p1_3d, cameraMatrix, dist)
            p2 = convert2D(p2_3d, cameraMatrix, dist)
            cv.line(frame, p1.astype(int), p2.astype(int), (255, 107, 170), 2)
            
            targets = checkBlocks(blocks, blockStatus, cameraMatrix, dist, depth, cone, frame, self.shift, True)

            descriptions = []
            for t in targets:
                descriptions.append(t.description)

            self.logger.append_csv(frame_count, body_id, descriptions)
