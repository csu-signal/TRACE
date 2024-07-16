from featureModules.IFeature import *
import numpy as np
from utils import convert2D, ConeShape, checkBlocks, Joint
import cv2 as cv

class GazeBodyTrackingFeature(IFeature):
    def __init__(self, shift):
        self.shift = shift

    def world_to_camera_coords(self, r_w, rotation, translation):
        return np.dot(rotation, r_w) + translation

    def get_joint(self, joint, body, rotation, translation):
        r_w = np.array(body["joint_positions"][joint.value])
        return self.world_to_camera_coords(r_w, rotation, translation)

    def processFrame(self, bodies, w, h, rotation, translation, cameraMatrix, dist, frame, framergb, depth, blocks, blockStatus):
        for b in bodies:
            # for i in (Joint.HEAD, Joint.NECK, Joint.NOSE, Joint.EYE_LEFT,Joint.EAR_LEFT,Joint.EYE_RIGHT,Joint.EAR_RIGHT):
            #     pos = convert2D(b["joint_positions"][i.value], rotation, translation, cameraMatrix, dist)
            #     cv.circle(frame, pos.astype(int), 5, (255,255,255), -1)
            nose = self.get_joint(Joint.NOSE, b, rotation, translation)
            head = self.get_joint(Joint.HEAD, b, rotation, translation)

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
            cv.line(frame, p1.astype(int), p2.astype(int), (255,255,0), 2)
            
            checkBlocks(blocks, blockStatus, cameraMatrix, dist, depth, cone, frame, self.shift, True)
