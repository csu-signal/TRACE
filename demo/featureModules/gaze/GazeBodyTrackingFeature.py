import cv2 as cv
import numpy as np
import csv ###added

from demo.featureModules.IFeature import *
from demo.logger import Logger
from demo.featureModules.utils import ConeShape, Joint, checkBlocks, convert2D
from skimage.draw import line ###added
import json ###added
from tqdm import tqdm

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
    
    ###w, h, framergb not used; depth, blocks, blockStatus, frame_count not used by hack to get joint depth data for ML algorithm
    ###processFrame is used for pre-recorded .mkv files
    def processFrame(self, bodies, w, h, rotation, translation, cameraMatrix, dist, frame, framergb, depth, blocks, blockStatus, frame_count):
        print("in process frame", frame)
        for b in bodies:
            body_id = b["body_id"]
            ###nose/ear_left/ear_right/eye_left/eye_right are all in camera coords as .get_joint converts from world coords to camera coords. Joint.xxxx are world coords
            nose = self.get_joint(Joint.NOSE, b, rotation, translation)

            ear_left = self.get_joint(Joint.EAR_LEFT, b, rotation, translation)
            ear_right = self.get_joint(Joint.EAR_RIGHT, b, rotation, translation)
            ear_center = (ear_left + ear_right) / 2

            eye_left = self.get_joint(Joint.EYE_LEFT, b, rotation, translation)
            eye_right = self.get_joint(Joint.EYE_RIGHT, b, rotation, translation)
            ###print("Joint.NOSE",b["joint_positions"][Joint.NOSE.value])
            ###print("body_id",body_id,"\nnose",nose,"\near_left",ear_left,"\near_right",ear_right,"\neye_left",eye_left,"\neye_right",eye_right)
            dir = nose - ear_center
            dir /= np.linalg.norm(nose - ear_center)
            origin = (eye_left + eye_right + nose) / 3
            ###p1_3d/p2_3d are in camera coords
            p1_3d = origin
            p2_3d = origin + 1000*dir
            ###print("origin",origin,"point2",p2_3d)
            cone = ConeShape(p1_3d, p2_3d, 80, 100, cameraMatrix, dist)
            cone.projectRadiusLines(self.shift, frame, False, False, True)
            ###convert2D returns pixel coordinates in column, row format rather than row, column format
            p1 = convert2D(p1_3d, cameraMatrix, dist)
            p2 = convert2D(p2_3d, cameraMatrix, dist)
            cv.line(frame, p1.astype(int), p2.astype(int), (255, 107, 170), 2)
            ###print("p1",p1,"p2",p2,"\n")
            targets = checkBlocks(blocks, blockStatus, cameraMatrix, dist, depth, cone, frame, self.shift, True)
            descriptions = []
            for t in targets:
                descriptions.append(t.description)
            ###jm code
            ###print("just before loop")
            ###Get the depth for the nose/eyes/ears pixels in camera coords as targets for the ML algorithm. 
            ###Get the nose/eyes/ears pixel coordinate + the gaze line in pixel coordinates as input to the ML agorithm.
            ###This function translates nose/eyes/ears and the gaze line in camera coords 3D locations to pixel locations.
            ###It returns the depth at each of the nose/eyes/ears joint location in camera coords.
            ###def _depthPixels(joint_tuple_of_camera_3D_coords, p1, p2, cameraMatrix, dist):
            nose_eyes_ears_depth_camera_coords = []
            nose_eyes_ears_depth_gazeLine_pixels = []
            for joint in (nose,eye_left,eye_right,ear_left,ear_right): ###joint_tuple_of_camera_3D_coords:
                nose_eyes_ears_depth_camera_coords.append(joint[2])
                ###print("joint[2] (depth)",joint[2])
                joint_2D = convert2D(joint, cameraMatrix, dist)
                if int(joint_2D[0]) > 1920 or int(joint_2D[1]) > 1080:
                    print("int(joint_2D[0]),int(joint_2D[1])",int(joint_2D[0]),int(joint_2D[1]))
                ###print("pixel location of joint",joint_2D)
                nose_eyes_ears_depth_gazeLine_pixels.append([int(joint_2D[0]),int(joint_2D[1])])
            ###print("nose_eyes_ears_depth_gazeLine_pixels",nose_eyes_ears_depth_gazeLine_pixels)
            p2_3d_short = origin + 100*dir
            p2_short = convert2D(p2_3d_short, cameraMatrix, dist)
            ###Get the gaze line as pixel coordinates
            rr, cc = line(int(p1[0]),int(p1[1]),int(p2_short[0]),int(p2_short[1])) ###This function inputx x0,y0,x1,y1 and calculates the pixels that represent the line between these two endpoints
            ###print("len rr",len(rr))
            for i in range(len(rr)):
                nose_eyes_ears_depth_gazeLine_pixels.append([rr[i],cc[i]])
            ###    return nose_eyes_ears_depth_camera_coords, nose_eyes_ears_depth_gazeLine_pixels
            ###nose_eyes_ears_depth_camera_coords, nose_eyes_ears_depth_gazeLine_pixels = _depthPixels((nose,eye_left,eye_right,ear_left,ear_right), p1, p2, cameraMatrix, dist)
            ###print(nose_eyes_ears_depth_camera_coords,'\n',nose_eyes_ears_depth_gazeLine_pixels)
            with open('depth_joints.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow([nose_eyes_ears_depth_gazeLine_pixels,nose_eyes_ears_depth_camera_coords])
            self.logger.append_csv(frame_count, body_id, descriptions)
            
    ###this function is used to process master-skeleton.json files for Groups01 - 10
    def processMasterSkeleton(self, rotation, translation, cameraMatrix, dist, master_skeleton_file):
        ###with open('C:/Users/jimmu/Desktop/CS793/Weights Task Dataset/Group_01/Group_01-master-skeleton.json', 'r') as f:
        with open(master_skeleton_file, 'r') as f:
        # Load the JSON data into a Python dictionary
            data = json.load(f) ### all are in camera coords as .get_joint converts from world coords to camera coords
            ###for frame in tqdm(range(0,len(data['frames']),5)): ###every 5 frames
            for frame in tqdm(range(0,len(data['frames']))): ###all frames
                for b in range(data['frames'][frame]['num_bodies']):
                    ###nose/ear_left/ear_right/eye_left/eye_right are all in 
                    nose = self.world_to_camera_coords(np.array(data['frames'][frame]['bodies'][b]['joint_positions'][27]), rotation, translation)
                    ear_left = self.world_to_camera_coords(np.array(data['frames'][frame]['bodies'][b]['joint_positions'][29]), rotation, translation)
                    ear_right = self.world_to_camera_coords(np.array(data['frames'][frame]['bodies'][b]['joint_positions'][31]), rotation, translation)
                    ear_center = (ear_left + ear_right) / 2
                    eye_left = self.world_to_camera_coords(np.array(data['frames'][frame]['bodies'][b]['joint_positions'][28]), rotation, translation)
                    eye_right = self.world_to_camera_coords(np.array(data['frames'][frame]['bodies'][b]['joint_positions'][30]), rotation, translation)
                    ###print("body_id",body_id,"\nnose",nose,"\near_left",ear_left,"\near_right",ear_right,"\neye_left",eye_left,"\neye_right",eye_right)
                    dir = nose - ear_center
                    dir /= np.linalg.norm(nose - ear_center)
                    origin = (eye_left + eye_right + nose) / 3
                    ###Get the depth for the nose/eyes/ears pixels in camera coords as targets for the ML algorithm. 
                    ###Get the nose/eyes/ears pixel coordinate + the gaze line in pixel coordinates as input to the ML agorithm.
                    ###This function translates nose/eyes/ears and the gaze line in camera coords 3D locations to pixel locations.
                    ###It returns the depth at each of the nose/eyes/ears joint location in camera coords. Note that the gazeline
                    ###data collected here was eventually not used as that is the actual end goal for the prediction.
                    nose_eyes_ears_depth_camera_coords = []
                    nose_eyes_ears_depth_gazeLine_pixels = []
                    for joint in (nose,eye_left,eye_right,ear_left,ear_right): ###joint_tuple_of_camera_3D_coords:
                        nose_eyes_ears_depth_camera_coords.append(joint[2])
                        ###print("joint[2] (depth)",joint[2])
                        joint_2D = convert2D(joint, cameraMatrix, dist)
                        ###check if within the field of view of the azure kinect camera
                        if int(joint_2D[0]) > 1920 or int(joint_2D[1]) > 1080:
                            print("int(joint_2D[0]),int(joint_2D[1])",int(joint_2D[0]),int(joint_2D[1]))
                        ###print("pixel location of joint",joint_2D)
                        nose_eyes_ears_depth_gazeLine_pixels.append([int(joint_2D[0]),int(joint_2D[1])])
                    ###print("nose_eyes_ears_depth_gazeLine_pixels",nose_eyes_ears_depth_gazeLine_pixels)
                    origin_2d = convert2D(origin, cameraMatrix, dist)
                    p2_3d_short = origin + 100*dir ###a shortened gazeline was originally used so that it didn't go outside the bounding box
                    p2_short = convert2D(p2_3d_short, cameraMatrix, dist)
                    ###Get the gaze line as pixel coordinates
                    ###This function inputx x0,y0,x1,y1 and calculates the pixels that represent the line between these two endpoints
                    rr, cc = line(int(origin_2d[0]),int(origin_2d[1]),int(p2_short[0]),int(p2_short[1])) 
                    ###print("len rr",len(rr))
                    for i in range(len(rr)):
                        nose_eyes_ears_depth_gazeLine_pixels.append([rr[i],cc[i]])
                    ###    return nose_eyes_ears_depth_camera_coords, nose_eyes_ears_depth_gazeLine_pixels
                    ###nose_eyes_ears_depth_camera_coords, nose_eyes_ears_depth_gazeLine_pixels = _depthPixels((nose,eye_left,eye_right,ear_left,ear_right), p1, p2, cameraMatrix, dist)
                    ###print(nose_eyes_ears_depth_camera_coords,'\n',nose_eyes_ears_depth_gazeLine_pixels)
                    ###with open('depth_joints_from_json_Group_01_to_Group_09_all_frames.csv', 'a') as f:
                    '''commented out to get actual_gaze_line_by_frame_camera_coords 
                    with open('depth_joints_from_json_Group10_all_frames.csv', 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([nose_eyes_ears_depth_gazeLine_pixels,nose_eyes_ears_depth_camera_coords])
                    ###this next section gets the actual nose/ear coords for group10 (set in __main__.py), which in depth_est_bignet_ablation2_test_Group10 has
                    ###the depth coordinate replaced by the prediction so that a comparison can be made on the cosine similarity
                    with open('nose_ear_left_ear_right_by_frame_camera_coords.csv', 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([nose,ear_left,ear_right])
                    '''
                    ###this next section saves the origin and direction
                    with open('origin_dir_by_frame_camera_coords.csv', 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([origin, dir])